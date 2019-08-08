# Master Thesis -- Process Extraction for Biorefinery Texts

## Code for the Master thesis project 
### Text Mining Pipeline for Process Extraction
The pipeline for process extraction constructed is as follows: 
- **Retrieve Domain-specific Articles** 
- **Select out the text for analysis**
- **Preprocess the dataset** (Noise removement, Text segmentation, Tokenization, PoS Tagging, Noun-Phrase Chunking, etc.)
- **Entity extraction**
  - Baseline method (Dictionary-based Look-up method)
  - Distributional semantics-based method (Utilize word embedding models to find similar terms)
- **Process Extraction**
  - A process, as defined in this work, can be extracted if a triple (input, technology, output) can be extracted from one sentence. 


***The code is pretty messy and it needs improvement, thus, I'm reading books about how to organize the code better. Under this situation, I do think it's better to contact me directly if you have questions towards the this project.***


