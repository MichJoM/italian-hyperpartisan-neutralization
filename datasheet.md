# PartisanLens Dataset


## Motivation


- **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*
    
The dataset was created to enable research on rephrasing hyperpartisan content and analyse how LLMs and other architectures respond to the task.

- **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
    
  The dataset was created by XXX, XXX, XXX, XXX.


- **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*
    
We will disclose the fundings upon acceptance.

- **Any other comments?**
    
    None.





## Composition

- **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*
    
  The instances are article_id from the original Semeval 2023 task 3 dataset, edited and neutralized version of the paragraph (human gold standard). Researchers can match the original ids, follow our prompt to recreate the entire parallel corpus.


- **How many instances are there in total (of each type, if appropriate)?**
    
  There are 825 unique instances in the original Par-ITA parallel corpus.


- **Does the dataset contains all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*
    
  It is a subset of Semeval 2023 task 3, containing ~2.5k instances of italian news paragraphs. The sampled dataset was curated trough manual annotation for hyperpartisan/neutral classification and considering linguistic cues typical of hyperpartisan language.


- **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*
    
  Each instance consists of the article_id (to retrieve the original hyperpartisan paragraph), human edited neutralized paragraph, an entry_id for our analaysis.  

- **Is there a label or target associated with each instance? If so, please provide a description.**
    
     Our edited_rewritten_text column represents the target. See above to how retrieve the original instance.

- **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*
    
  Everything is included. No data is missing.


- **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *( If so, please describe how these relationships are made explicit.)*
    
  No, each instance is isolated from the others and no relationships exist and if so, it was not made explicit.


- **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*
    
  In our experiments, we dedicated a random 20% of the data for testing. We recommend a similar split for future training.


- **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*
    
  Potential errors in the annotation process might be carried from each original dataset author.


- **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*
    
  The dataset is self-contained. It was created using external resources, Semeval 2023 task 3. The authors stated that: "The dataset may include content which is protected by copyright of third parties. It may only be used in the context of this shared task, and only for scientific research purposes. The dataset may not be redistributed or shared in part or full with any third party. You may not share you passcode with others or give access to the dataset to unauthorised users. Any other use is explicitly prohibited.". We are not sharing/redistributing the original data. We used this dataset for research purpose only. 


- **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*
    
  Unknown to the authors of the datasheet. 


- **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*
    
  Yes, because it is a hyperpartisan dataset, so data contains explicit hate speech-language against immigrant policies or data that might cause harm.


- **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
 Not directly. Some politicians and institutions are cited due to the nature of the dataset. 

- **Does the dataset identify any subpopulations (e.g., by age, gender)?** *(If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.)*
    
  This is not explicitly identified, though some of the paragraphs might mention the country of origin, gender or other characteristics of some people.


- **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** *(If so, please describe how.)*
    
  Yes, individuals and institutions are explicitly mentioned in the headlines.


- **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** *(If so, please provide a description.)*
    
   Sensitive data is not explicitly identified, though some of the comments might mention any information of some people.


- **Any other comments?**
    
  None.




## Collection Process


- **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*
    
  The data was collected by accessing to Semeval 2023 Task 3 repository via the request form: https://propaganda.math.unipd.it/semeval2023task3/registration.php. 

- **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*

  See above. 


- **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
    
  NA.


- **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
    
  All the data collection was done by the authors.


- **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*
   
   From 2020 to mid 2022.


- **Were any ethical review processes conducted (e.g., by an institutional review board)?** *(If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
  No.    


- **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
 
  Yes, it is about people, politicians and institutions on news articles, where hyperpartisan messages may occur.    


- **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**

  We obtained the data using the Registration Form provided by the Semeval 2023 task 3 organizers.    


- **Were the individuals in question notified about the data collection?** *(If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.)*
    
  We suppose so, since the request was officially made through the provided platform (see above). 

- **Did the individuals in question consent to the collection and use of their data?** *(If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.)*
    
   Yes, consent was managed through am approach involving the original dataset source and the specific experts recruited for this study.

    Original Source Data (SemEval-2023): The base content was retrieved from the SemEval-2023 Task 3 dataset. Use of this data was authorized under the task's specific scientific research license. As this is a public benchmark derived from online news media, individual consent for the original authors is typically handled under the "Fair Use" or "Scientific Research" provisions of the source license.

- **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** *(If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).)*
    
  NA.

- **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** *(If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
  No.

- **Any other comments?**

  None.





## Preprocessing/cleaning/labeling


- **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*

  Yes, we filtered the interested instances after labeling the italian paragraphs.    


- **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*

  Yes. The raw data can be accessed via the link provided above.    


- **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*

  No. We could release our annotation platforms upon request and for research purpose only.   


- **Any other comments?**
  
  None.   





## Uses


- **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*

    At the time of the submission, only the original paper (Semeval 2023 task 3).


- **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*
    
  No.


- **What (other) tasks could the dataset be used for?**

  The dataset could be used for anything related to modelling or understanding hyperpartisan, on italian paragraphs about immigration, Ukraine invasion, Covid, European Politics in EU.


- **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** *(For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)  If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?)*

NA.

- **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*
 
  Yes, the dataset should not be used for tasks that promote hyperpartisan, offensive and conspiracionist language and censorship towards individuals or communities. Furthermore, the dataset must not be used to censor the journalist.   


- **Any other comments?**

  None




## Distribution


- **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*
    
  Yes, the dataset is freely available under Semeval 2023 task 3 and ours conditions. 


- **How will the dataset will be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*
 
   We are evaluating on how to distribute it.


- **When will the dataset be distributed?**
 
   The dataset will be distributed from XX.


- **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*
    
NA.

- **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*
 
  Not to our knowledge.   


- **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*
  
  Yes, some instances of the dataset may need, apart from our terms and conditions, the original terms and conditions of the original dataset.     


- **Any other comments?**
  
  None.   





## Maintenance


- **Who is supporting/hosting/maintaining the dataset?**
 
  XX is supporting/maintaining the dataset.     


- **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
 
  Via email XX   


- **Is there an erratum?** *(If so, please provide a link or other access point.)*
    
  Currently, no. As errors are encountered, future versions of the dataset may be released (versioned). They will all be provided in the same GitHub location.


- **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*
    
  Same as previous.


- **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** *(If so, please describe these limits and explain how they will be enforced.)*

  No.


- **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*
 
  Yes, all data will be versioned.   


- **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*
 
   Feedback may be submitted via issues on GitHub. 


- **Any other comments?**

  None.    