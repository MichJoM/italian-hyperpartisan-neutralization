"""Custom Seq2Seq trainer with classifier-guided loss for neutrality."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from transformers import Seq2SeqTrainer


logger = logging.getLogger(__name__)


class NeutralityGuidedTrainer(Seq2SeqTrainer):
    """Seq2Seq trainer with neutrality classifier guidance during training.

    This trainer extends the standard Seq2SeqTrainer by periodically generating
    sequences during training and using a neutrality classifier to compute an
    additional loss component that guides the model toward more neutral outputs.

    Args:
        neutrality_classifier: Instance of NeutralityClassifier to score generations
        guidance_weight: Weight for the neutrality guidance loss component (default: 1.0)
        guidance_every_n_steps: Apply guidance every N training steps (default: 5)
        guidance_target_rate: Target neutrality rate, penalties apply below this (default: 0.7)
        guidance_max_length: Max length for guidance generations (default: 128)
        *args, **kwargs: Passed to Seq2SeqTrainer
    """

    def __init__(
        self,
        *args,
        neutrality_classifier=None,
        guidance_weight: float = 1.0,
        guidance_every_n_steps: int = 5,
        guidance_target_rate: float = 0.7,
        guidance_max_length: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.neutrality_classifier = neutrality_classifier
        self.guidance_weight = float(guidance_weight)
        self.guidance_every_n_steps = int(max(1, guidance_every_n_steps))
        self.guidance_target_rate = float(guidance_target_rate)
        self.guidance_max_length = int(guidance_max_length)

        # Freeze classifier to prevent training
        if self.neutrality_classifier is not None:
            self.neutrality_classifier.model.eval()
            for param in self.neutrality_classifier.model.parameters():
                param.requires_grad = False
            logger.info(
                "Neutrality-guided training enabled: weight=%.2f, every_n=%d, target=%.2f",
                self.guidance_weight,
                self.guidance_every_n_steps,
                self.guidance_target_rate,
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # ← Add this line (ignore the value)
    ):
        # Standard seq2seq loss (always computed)
        outputs = model(**inputs)
        seq2seq_loss = outputs.loss

        # Determine if we should apply guidance this step
        should_guide = (
            self.neutrality_classifier is not None
            and model.training  # Only during training, not eval
            and self.state.global_step % self.guidance_every_n_steps == 0
            and self.guidance_weight > 0
        )

        if should_guide:
            try:
                neutrality_loss = self._compute_neutrality_loss(model, inputs)
                total_loss = seq2seq_loss + self.guidance_weight * neutrality_loss

                # Log metrics periodically
                if self.state.global_step % self.args.logging_steps == 0:
                    self.log(
                        {
                            "train_seq2seq_loss": seq2seq_loss.item(),
                            "train_neutrality_guidance_loss": neutrality_loss.item(),
                            "train_total_loss": total_loss.item(),
                        }
                    )
            except Exception as e:
                # Fallback to standard loss if guidance fails
                logger.warning(
                    "Neutrality guidance failed at step %d: %s. Using standard loss.",
                    self.state.global_step,
                    str(e),
                )
                total_loss = seq2seq_loss
        else:
            total_loss = seq2seq_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_neutrality_loss(self, model, inputs) -> torch.Tensor:
        """Compute neutrality-based loss component by generating and scoring text."""
        # Unwrap DDP to access .generate()
        unwrapped_model = model.module if hasattr(model, "module") else model

        # Generate sequences with current model state (no gradients needed)
        gen_kwargs = {
            "input_ids": inputs.get("input_ids"),
            "max_length": self.guidance_max_length,
            "num_beams": 1,
            "do_sample": False,
            "pad_token_id": self.processing_class.pad_token_id,  # Updated for deprecation
            "eos_token_id": self.processing_class.eos_token_id,
        }
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs["attention_mask"]
        with torch.no_grad():
            generated_ids = unwrapped_model.generate(**gen_kwargs)

        # Decode generated sequences (use processing_class for future compatibility)
        generated_texts = self.processing_class.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Score with neutrality classifier
        neutrality_scores = self._score_neutrality(generated_texts)

        # Compute loss: penalize falling short of target rate
        mean_neutrality = neutrality_scores.mean()
        shortfall = torch.clamp(self.guidance_target_rate - mean_neutrality, min=0.0)

        return shortfall

    def _score_neutrality(self, texts: list[str]) -> torch.Tensor:
        """Score texts with the neutrality classifier.

        Args:
            texts: List of text strings to score

        Returns:
            torch.Tensor: Neutrality probability for each text [batch_size]
        """
        if not texts:
            return torch.tensor([0.0], device=self.args.device)

        # Encode texts for classifier
        encoded = self.neutrality_classifier.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.neutrality_classifier.max_length,
            return_tensors="pt",
        ).to(self.neutrality_classifier.device)

        # Get classifier predictions (no gradients)
        with torch.no_grad():
            logits = self.neutrality_classifier.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            neutral_probs = probs[:, self.neutrality_classifier.neutral_id]

        # Return on training device
        return neutral_probs.to(self.args.device)
