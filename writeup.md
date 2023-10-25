# AI-Finder Results Report
### Jeremy Hadfield | May 29, 2022

- Google Colab Link: [ought_project.ipynb](https://colab.research.google.com/drive/1CYSTGYtmu0Jiw_0BDkjM7CpohQ_xi5D1?usp=drive_link) (contains all code for the project)
- Predictions File: [predictions.jsonl](https://drive.google.com/file/d/1jHcNP8MJn-eJ3s6WxcpUeWRFyeJCCT1i/view?usp=drive_link) (contains all predictions for the unlabeled test data)
- Google docs writeup version (better formatting): [Ought Project Writeup](https://docs.google.com/document/d/1CYlvHClIYJAAr0xxQuIF9HoIqUu4jfl17l_beMoGYbM/edit?usp=drive_link)
- Total time taken on the project: 10 hours (including writeup time)

## Few-Shot Learning with GPT-2
**Classification accuracy**: 40-50% (depending on random sample & prompt)

I started by initializing a Dataset class to read in the datasets. Then, I constructed a base class for the GPT-2 language model (GPT2LM) and a GPTClassifier class. To classify papers, I prompted the model with the instructions and a few (3-4) examples. 

The GPT-2 model did not achieve stellar performance on this few-shot text classification task. Due to limitations in Google Colab’s capacity, I had to evaluate the model’s performance using a random uniform sample of 50 items (containing an equal amount of ‘AI’ and ‘Not AI’ papers) rather than the entire dataset. I evaluated the model several times, tinkering with the prompt and the examples each time. Every time, the model’s accuracy was between 40% and 50%. 

However, GPT-2 only achieved an accuracy of 45.8% on the RAFT benchmark tests for text classification. Even GPT-3 only reached an accuracy of 62.7%, where human participants achieved 73.5% accuracy. Therefore, GPT-2’s performance on this task is standard and in line with our prior expectations. 

Problems: More optimizations could likely improve the model’s accuracy further, but are unlikely to yield significant improvements. I initially had a lot of problems with exceeding the max sequence length for GPT-2 (1024), as the prompts are very long. To fix this issue, I had to truncate the input tensors — but this reduces the model’s accuracy. Further, I was unable to use any more than 4 examples without overflowing the max sequence length. 

### Possible next steps to improve accuracy: 
- Batch encoding of the prompts could allow us to use more examples without truncating the input sequences, giving the model more information.
- Using larger GPT-2 models from HuggingFace like gpt2-large or gpt2-xl could improve the model’s performance and allow using more examples without truncation. 
- Tweaking the instructions (prompt engineering) could yield slight improvements.
  
## Keyword-Based Classification
**Classification accuracy**: 91.6% (on test data) and ​​92.0% (on dev data). 

This simple approach classifies papers as AI-relevant based on the presence of certain keywords. This simulates how a human might classify papers - by scanning for keywords they recognize in the titles and abstracts. To create this model, I manually created a shortlist of 15 keywords that I expected to be used far more in AI-relevant papers than in other scientific papers. While some of these keywords were my educated guesses, I chose others from the most frequent words in the AI-relevant labeled examples from the given datasets. The model classifies papers as AI-relevant if the text (title + abstract) contains any of these keywords. 

Somewhat surprisingly, this model achieved an accuracy of over 90%, far more than most few-shot text classification tasks. This demonstrates the effectiveness of a naive keyword-based approach that leverages domain-specific knowledge about artificial intelligence. Arguably, it also qualifies as few-shot, since it does not require any examples and only uses 15 keywords. This model is also much more lightweight than a full-scale language model and runs far faster. For relatively simple tasks like this one, it may be more effective to just use a keyword dictionary rather than training a computationally expensive Transformer model. 

Problems: Low generalizability and overfitting might be concerns for this approach, as this static list of keywords may be over-represented in this dataset compared to the full corpus of all AI-relevant papers. Further, there may be papers not relevant to AI that use these keywords, resulting in false positives.However, it is a reasonable assumption that most papers that contain these technical and domain-specific keywords are actually relevant to artificial intelligence. 

Possible next steps: 
Consult with more AI researchers and browse more papers to identify better keywords. 
Use a more computational and less manual technique to identify keywords, calculating the density of these keywords in datasets of AI-relevant and AI-irrelevant papers. 
Classify results as AI-relevant based on a composite score calculated from the frequency of the keywords in the text, not just the presence of any one of these keywords. 

## Zero-Shot Topic Classification with BART
**Classification accuracy**: 98.0% (hypothesis method) or 78.0% (topic classification method). 

Finally, I tried using zero-shot learning to classify the papers without any examples. I used two methods here, both involving the BART Transformer-based autoencoder (using the HuggingFace bart-large-mnli pretrained model). The first method treats the text as a premise and tests the hypothesis that the text is about artificial intelligence. If the probability that the premise entails the hypothesis, exceeds a threshold (60%), it classifies the paper as AI-relevant. The second method classifies the text into a list of overarching topics, compiled from the ArXiV taxonomy of paper categories. If the highest-scoring topic is AI, it classifies the paper as AI-relevant. 

Problems: Both of these models (especially the hypothesis method) achieve remarkably high accuracy, and as these are zero-shot learning methods, they are likely generalizable to out-of-sample data. However, they are both quite slow to run and evaluate, and they require using a relatively heavyweight language model. 

Possible next steps: 
Tinker with the hypothesis text and the topic list to see if changes can improve accuracy. 
Combine multiple zero-shot models (perhaps including the two methods described here), classifying papers based on an aggregate of the predictions from all of the models. 

Recommendation: I would recommend using the zero-shot BART model with the hypothesis-testing method to classify papers as AI-relevant. This achieves 90%+ accuracy, does not rely on any examples, runs relatively quickly, and will almost certainly generalize to out-of-sample data. Further, it could be useful to combine this method with the keyword-based model, multiplying the probability that the premise entails the hypothesis by some score based on the density of the keywords in the text. 

## Next Steps
Using similarity score with GPT-2: Another way to classify papers as AI-relevant is to compare the final-layer embeddings of the examples and the text. The similarity score between these embeddings (computed with matrix multiplication or the dot product) indicates the likelihood the text is AI-relevant. This does not require a complete prompt, and can be a faster and potentially more accurate way to classify the texts. 
Text generation to expand the initial dataset for few-shot learning: Given only 20 examples, it is hard to fine-tune models like GPT-2, and they have to make predictions based on very limited information. Feeding these limited examples into a language model like GPT-2 to generate more examples could expand the sample-set and improve accuracy. Further, these methods are often still generalizable. 
Using GPT-2 to Create Synthetic Data to Improve the Prediction Performance of NLP Machine Learning Classification Models
Leveraging GPT-2 for Classifying Spam Reviews with Limited Labeled Data via Adversarial Training
Alternative machine learning models: GPT-2 is somewhat outdated, and there are several other models that would likely outperform it for few-shot classification. 
T-Few model - the current best-performing model for few-shot text classification, better accuracy than humans & GPT-3 on the RAFT dataset (source).
Defined in paper Liu et al (2022) - Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
Github repository - https://github.com/r-three/t-few
DistilBERT - a distilled version of BERT: it has 40% fewer parameters, runs 60% faster while preserving 97% of BERT's performance as measured on the GLUE language understanding benchmark.
How Hugging Face achieved a 2x performance boost for Question Answering with DistilBERT in Node.js — The TensorFlow Blog
LSTMs - long short-term memory recurrent neural networks have been successful at few-shot text classification tasks, although they may not outperform GPT. 
A C-LSTM Neural Network for Text Classification 
LSTM for Text Classification in Python
