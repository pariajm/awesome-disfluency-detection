# Awesome Disfluency Detection
A curated list of awesome disfluency detection publications along with their released code (if available) and bibliography. A chronological order of the published papers is available [here](https://github.com/pariajm/awesome-disfluency-detection/tree/chronologically-ordered-papers).

## Contributing
Please feel free to send me pull requests or [email me](#contact) to add a new resource.

## Table of Contents
- [Papers](#papers)
   - [Noisy Channel Models](#noisy-channel-models)
   - [Sequence Tagging Models](#sequence-tagging-models)
   - [Translation Based Models](#translation-based-models)
   - [Using Acoustic/Prosodic Cues](#using-acousticprosodic-cues)
   - [Data Augmenatation Techniques](#data-augmenatation-techniques)
   - [Incremental Disfluency Detection](#incremental-disfluency-detection)
   - [E2E Speech Recognition and Disfluency Removal](#e2e-speech-recognition-and-disfluency-removal)
   - [E2E Speech Translation and Disfluency Removal](#e2e-speech-translation-and-disfluency-removal)
   - [Others](#others)
   
- [Theses](#theses)
- [Contact](#contact)

# Papers
Studies on disfluency detection are categorized as follows (some papers belong to more than one category):

### Noisy Channel Models
*The main idea behind a noisy channel model of speech disfluency is that we assume there is a fluent source utterance `x` to which some noise has
been added, resulting in a disfluent utterance `y`. Given `y`, the goal is to find the most likely source fluent sentence such that `p(x|y)` is maximized.*

 * [Disfluency detection using a noisy channel model and a deep neural language model.](https://www.aclweb.org/anthology/P17-2087.pdf) Jamshid Lou *et al.* ACL 2017. [[bib]](https://www.aclweb.org/anthology/P17-2087.bib)

 * [The impact of language models and loss functions on repair disfluency detection.](https://www.aclweb.org/anthology/P11-1071.pdf) Zwarts *et al.* ACL 2011. [[bib]](https://www.aclweb.org/anthology/P11-1071.bib)

 * [An improved model for recognizing disfluencies in conversational speech.](http://web.science.mq.edu.au/~mjohnson/papers/rt04.pdf) Johnson *et al.* Rich Transcription Workshop 2004. [[bib]](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.6045)

 * [A TAG-based noisy channel model of speech repair.](https://www.aclweb.org/anthology/P04-1005.pdf) Johnson *et al.* ACL 2004. [[bib]](https://www.aclweb.org/anthology/P04-1005.bib)

### Sequence Tagging Models
*The task of disfluency detection is framed as a word token classification problem, where each word token is classified as being disfluent/fluent or by using a begin-inside-outside (BIO) based tagging scheme.*

* [Joint prediction of punctuation and disfluency in speech transcripts.](http://www.interspeech2020.org/uploadfile/pdf/Mon-2-5-9.pdf) Lin *et al*. INTERSPEECH 2020. [[bib]](https://isca-speech.org/archive/Interspeech_2020/abstracts/1277.html)

* [Giving attention to the unexpected: using prosody innovations in disfluency detection.](https://www.aclweb.org/anthology/N19-1008.pdf) Zayats *et al.* NAACL 2019. [[bib]](https://www.aclweb.org/anthology/N19-1008.bib) [[code]](https://github.com/vickyzayats/disfluency_detection)

* [Disfluency detection based on speech-aware token-by-token sequence labeling with BLSTM-CRFs and attention mechanisms.](http://www.apsipa.org/proceedings/2019/pdfs/185.pdf) Tanaka *et al.* APSIPA 2019. [[bib]](https://dblp.org/rec/conf/apsipa/TanakaMMOA19.html?view=bibtex)

* [Noisy BiLSTM-based models for disfluency detection.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1336.pdf) Bach *et al.* INTERSPEECH 2019.  [[bib]](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1336.html)

* [Disfluency detection using auto-correlational neural networks.](https://www.aclweb.org/anthology/D18-1490.pdf) Jamshid Lou *et al.* EMNLP 2018. [[bib]](https://www.aclweb.org/anthology/D18-1490.bib) [[code]](https://github.com/pariajm/deep-disfluency-detector)

* [Robust cross-domain disfluency detection with pattern match networks.](https://arxiv.org/pdf/1811.07236.pdf) Zayats *et al.* Arxiv 2018. [[bib]](https://dblp.org/rec/journals/corr/abs-1811-07236.html?view=bibtex) [[code]](https://github.com/vickyzayats/disfluency_detection)

* [Disfluency detection using a bidirectional LSTM.](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/1247.PDF) Zayats *et al.* INTERSPEECH 2016. [[bib]](https://dblp.org/rec/conf/interspeech/ZayatsOH16.html?view=bibtex)

* [Multi-domain disfluency and repair detection.](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_2907.pdf) Zayats *et al.* INTERSPEECH 2014. [[bib]](https://www.bibsonomy.org/bibtex/27bbcc3f054360422115c081361473179/dblp)

* [A Sequential Repetition Model for Improved Disfluency Detection.](https://isca-speech.org/archive/archive_papers/interspeech_2013/i13_2624.pdf) Ostendorf *et al.* INTERSPEECH 2013. [[bib]](https://www.bibsonomy.org/bibtex/b67c92313f3cd785db73f0bf5cf1d8ca)

* [The role of disfluencies in topic classification of human-human conversations.](http://www.icsi.berkeley.edu/pubs/speech/enrichspeech.pdf) Liu *et al.* IEEE TRANSACTIONS ON SPEECH & AUDIO PROCESSING 2006. [[bib]](https://ieeexplore.ieee.org/document/1677974)

* [Automatic disfluency identification in conversational speech using multiple knowledge sources.](http://www.cs.columbia.edu/~julia/papers/liu03.pdf) Liu *et al.* Eurospeech 2003. [[bib]](https://www.semanticscholar.org/paper/Automatic-disfluency-identification-in-speech-using-Liu-Shriberg/d772b10d1a5ee70ee1daa9dccc66243a917c1b73)

* [Automatic punctuation and disfluency detection in multi-party meetings using prosodic and lexical cues.](https://www.isca-speech.org/archive/archive_papers/icslp_2002/i02_0949.pdf) Baron *et al.* ICSLP 2002. [[bib]](https://www.isca-speech.org/archive/icslp_2002/i02_0949.html)

### Translation Based Models
*Translation-based approaches for disfluency detection are commonly formulated as encoder-decoder systems, where the encoder learns the representation of input sentence containing disfluencies and the decoder learns to generate the underlying fluent version of the input.*

* [Adapting translation models for transcript disfluency detection.](https://www.aaai.org/ojs/index.php/AAAI/article/view/4597) Dong *et al.* AAAI 2019. [[bib]](https://ojs.aaai.org/index.php/AAAI/citationstylelanguage/download/bibtex?submissionId=4597&publicationId=3002)

* [Semi-supervised disfluency detection.](https://www.aclweb.org/anthology/C18-1299.pdf) Wang *et al.* COLING 2018. [[bib]](https://www.aclweb.org/anthology/C18-1299.bib)

* [A neural attention model for disfluency detection.](https://www.aclweb.org/anthology/C16-1027.pdf) Wang *et al.* COLING 2016. [[bib]](https://www.aclweb.org/anthology/C16-1027.bib)

### Parsing Based Models
*Parsing-based approaches detect disfluencies while simultaneously identifying the syntactic or semantic structure of the sentence. Training a parsing-based
model requires large annotated treebanks that contain both disfluencies and syntactic/semantic structures.*

* [Semantic parsing of disfluent speech.](https://assets.amazon.science/14/96/a1234b3941b98728d63540833193/semantic-partsing-of-disfluent-speech.pdf)
 Sen *et al.* EACL 2021. 
 
* [Improving disfluency detection by self-training a self-attentive model.](https://www.aclweb.org/anthology/2020.acl-main.346.pdf) Jamshid Lou *et al.* ACL 2020. [[bib]](https://www.aclweb.org/anthology/2020.acl-main.346.bib) [[code]](https://github.com/pariajm/joint-disfluency-detector-and-parser)

* [Neural constituency parsing of speech transcripts.](https://www.aclweb.org/anthology/N19-1282.pdf) Jamshid Lou *et al.* NAACL 2019. [[bib]](https://www.aclweb.org/anthology/N19-1282.bib) [[code]](https://github.com/pariajm/joint-disfluency-detector-and-parser/tree/naacl2019)

* [On the role of style in parsing speech with neural models.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3122.pdf) Tran *et al.* INTERSPEECH 2019. [[bib]](https://www.semanticscholar.org/paper/On-the-Role-of-Style-in-Parsing-Speech-with-Neural-Tran-Yuan/6658f850d2d7d4fa899bf2c8da93fc5ef1bd00b6) [[code]](https://github.com/trangham283/prosody_nlp/tree/master/code/self_attn_speech_parser)

* [Parsing speech: a neural approach to integrating lexical and acoustic-prosodic information.](https://www.aclweb.org/anthology/N18-1007.pdf) Tran *et al.* NAACL 2018. [[bib]](https://www.aclweb.org/anthology/N18-1007.bib) [[code]](https://github.com/shtoshni92/speech_parsing)

* [Transition-based disfluency detection using LSTMs.](https://www.aclweb.org/anthology/D17-1296.pdf) Wang *et al.* EMNLP 2017. [[bib]](https://www.aclweb.org/anthology/D17-1296.bib) [[code]](https://github.com/hitwsl/transition_disfluency)

* [Joint transition-based dependency parsing and disfluency detection for automatic speech recognition texts.](https://www.aclweb.org/anthology/D16-1109.pdf) Yoshikawa *et al.* EMNLP 2016. [[bib]](https://www.aclweb.org/anthology/D16-1109.bib)

* [Joint incremental disfluency detection and dependency parsing.](https://www.aclweb.org/anthology/Q14-1011.pdf) Honnibal *et al.* TACL 2014. [[bib]](https://www.aclweb.org/anthology/Q14-1011.bib)

* [Joint parsing and disfluency detection in linear time.](https://www.aclweb.org/anthology/D13-1013.pdf) Rasooli *et al.* EMNLP 2013. [[bib]](https://www.aclweb.org/anthology/D13-1013.bib)

* [Edit detection and parsing for transcribed speech.](https://www.aclweb.org/anthology/N01-1016.pdf) Charniak *et al.* NAACL 2001. [[bib]](https://www.aclweb.org/anthology/N01-1016.bib)

### Using Acoustic/Prosodic Cues
*Speech signal carries extra information beyond the words which can provide useful cues for disfluency detection models. Some studies have explored integrating acoustic/prosodic cues to lexical features for detecting disfluencies.*

* [On the role of style in parsing speech with neural models.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3122.pdf) Tran *et al.* INTERSPEECH 2019. [[bib]](https://www.semanticscholar.org/paper/On-the-Role-of-Style-in-Parsing-Speech-with-Neural-Tran-Yuan/6658f850d2d7d4fa899bf2c8da93fc5ef1bd00b6) [[code]](https://github.com/trangham283/prosody_nlp/tree/master/code/self_attn_speech_parser)

* [Disfluency detection based on speech-aware token-by-token sequence labeling with BLSTM-CRFs and attention mechanisms.](http://www.apsipa.org/proceedings/2019/pdfs/185.pdf) Tanaka *et al.* APSIPA 2019. [[bib]](https://dblp.org/rec/conf/apsipa/TanakaMMOA19.html?view=bibtex)

* [Giving attention to the unexpected: using prosody innovations in disfluency detection.](https://www.aclweb.org/anthology/N19-1008.pdf) Zayats *et al.* NAACL 2019. [[bib]](https://www.aclweb.org/anthology/N19-1008.bib) [[code]](https://github.com/vickyzayats/disfluency_detection)

* [Parsing speech: a neural approach to integrating lexical and acoustic-prosodic information.](https://www.aclweb.org/anthology/N18-1007.pdf) Tran *et al.* NAACL 2018. [[bib]](https://www.aclweb.org/anthology/N18-1007.bib) [[code]](https://github.com/shtoshni92/speech_parsing)

* [Automatic disfluency identification in conversational speech using multiple knowledge sources.](http://www.cs.columbia.edu/~julia/papers/liu03.pdf) Liu *et al.* Eurospeech 2003. [[bib]](https://www.semanticscholar.org/paper/Automatic-disfluency-identification-in-speech-using-Liu-Shriberg/d772b10d1a5ee70ee1daa9dccc66243a917c1b73)

* [Automatic punctuation and disfluency detection in multi-party meetings using prosodic and lexical cues.](https://www.isca-speech.org/archive/archive_papers/icslp_2002/i02_0949.pdf) Baron *et al.* ICSLP 2002. [[bib]](https://www.isca-speech.org/archive/icslp_2002/i02_0949.html)

### Data Augmenatation Techniques
*Disfluency detection models are usually trained and evaluated on Switchboard corpus. Switchboard is the largest disfluency annotated dataset; however, only about
6% of the words in the Switchboard are disfluent. Some studies have suggested new data augmentation techniques to mitigate the scarcity of gold disfluency-labeled data.*

* [Planning and generating natural and diverse disfluent texts as augmentation for disfluency detection.](https://www.aclweb.org/anthology/2020.emnlp-main.113.pdf) Yang *et al.* EMNLP 2020. [[bib]](https://www.aclweb.org/anthology/2020.emnlp-main.113.bib) [[code]](https://github.com/GT-SALT/Disfluency-Generation-and-Detection/tree/main/disfluency-detection)

* [Combining self-training and self-supervised learning for unsupervised disfluency detection.](https://www.aclweb.org/anthology/2020.emnlp-main.142.pdf) Wang *et al.* EMNLP 2020. [[bib]](https://www.aclweb.org/anthology/2020.emnlp-main.142.bib) [[code]](https://github.com/scir-zywang/self-training-self-supervised-disfluency)

* [Improving disfluency detection by self-training a self-attentive model.](https://www.aclweb.org/anthology/2020.acl-main.346.pdf) Jamshid Lou *et al.* ACL 2020. [[bib]](https://www.aclweb.org/anthology/2020.acl-main.346.bib) [[code]](https://github.com/pariajm/joint-disfluency-detector-and-parser) [[data]](https://github.com/pariajm/english-fisher-annotations)

* [Auxiliary sequence labeling tasks for disfluency detection.](https://arxiv.org/pdf/2011.04512v1.pdf) Lee *et al.* arxiv 2020. 

* [Multi-task self-supervised learning for disfluency detection.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj-kZrFrJPwAhXvyzgGHbPDBIEQFjAEegQIBBAD&url=https%3A%2F%2Faaai.org%2Fojs%2Findex.php%2FAAAI%2Farticle%2Fview%2F6456%2F6312&usg=AOvVaw0V6DdfU11M_WdzCrkWfLoo) Wang *et al.* AAAI 2020. [[bib]](https://dblp.org/rec/conf/aaai/WangCLQLW20.html?view=bibtex)

* [Noisy BiLSTM-based models for disfluency detection.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1336.pdf)  Bach *et al.* INTERSPEECH 2019. [[bib]](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1336.html)

* [Semi-supervised disfluency detection.](https://www.aclweb.org/anthology/C18-1299.pdf) Wang *et al.* COLING 2018. [[bib]](https://www.aclweb.org/anthology/C18-1299.bib)

### Incremental Disfluency Detection
*Most disfluency detection models are developed based on the assumptions that a full sequence context as well as rich transcriptions including pre-segmentation information are available. These assumptions, however, are not valid in real-time scenarios where the input to the disfluency detector is live transcripts generated by a streaming ASR model. In such cases, a disfluency detector is expected to incrementally label input transcripts as it receives token-by-token data. Some studies have proposed new incremental disfluency detectors.*

* [Re-framing incremental deep language models for dialogue processing with multi-task learning.](https://www.aclweb.org/anthology/2020.coling-main.43.pdf) Rohanian *et al.* COLING 2020. [[bib]](https://www.aclweb.org/anthology/2020.coling-main.43.bib) [[code]](https://github.com/mortezaro/mtl-disfluency-detection)

* [Recurrent neural networks for incremental disfluency detection.](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_0849.pdf) Hough *et al.* INTERSPEECH 2015. [[bib]](https://www.aclweb.org/anthology/Q14-1011.bib)

* [Joint incremental disfluency detection and dependency parsing.](https://www.aclweb.org/anthology/Q14-1011.pdf) Honnibal *et al.* TACL 2014. [[bib]](https://www.aclweb.org/anthology/Q14-1011.bib)

### E2E Speech Recognition and Disfluency Removal
*Most disfluency detectors are applied as an intermediate step between a speech recognition and a downstream task. Unlike the conventional pipeline models, some studies have explored end-to-end speech recoginition and disfluency removal.*

* [Improved robustness to disfluencies in RNN-Transducer based speech recognition.](https://assets.amazon.science/11/9c/1377940f42a58324c408b1017a2f/improved-robustness-to-disfluencies-in-rnn-transducer-based-speech-recognition.pdf) Mendelev *et al.* Arxiv 2020. [[bib]](https://dblp.org/rec/journals/corr/abs-2012-06259.html?view=bibtex)

* [End-to-end speech recognition and disfluency removal.](https://www.aclweb.org/anthology/2020.findings-emnlp.186.pdf) Jamshid Lou *et al.* EMNLP Findings 2020. [[bib]](https://www.aclweb.org/anthology/2020.findings-emnlp.186.bib) [[code]](https://github.com/pariajm/e2e-asr-and-disfluency-removal-evaluator)

### E2E Speech Translation and Disfluency Removal
*While most of the end-to-end speech translation studies have explored translating read speech, there are a few studies that examine the end-to-end conversational speech translation, where the task is to directly translate source disfluent speech into target fluent texts.*

* [NAIST’s machine translation systems for IWSLT 2020 conversational speech translation task.](https://www.aclweb.org/anthology/2020.iwslt-1.21.pdf) Fukuda *et al.* IWSLT 2020. [[bib]](https://www.aclweb.org/anthology/2020.iwslt-1.21.bib) 

* [Generating fluent translations from disfluent text without access to fluent references: IIT Bombay@IWSLT2020.](https://www.aclweb.org/anthology/2020.iwslt-1.22.pdf) Saini *et al.* IWSLT 2020. [[bib]](https://www.aclweb.org/anthology/2020.iwslt-1.22.bib) 

* [Fluent translations from disfluent speech in end-to-end speech translation.](https://www.aclweb.org/anthology/N19-1285.pdf) Salesky *et al.* NAACL 2019. [[bib]](https://www.aclweb.org/anthology/N19-1285.bib) [[data]](https://github.com/isl-mt/fluent-fisher)

* [Segmentation and disfluency removal for conversational speech translation.](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_0318.pdf) Hassan *et al.* INTERSPEECH 2014. [[bib]](https://www.microsoft.com/en-us/research/publication/segmentation-and-disfluency-removal-for-conversational-speech-translation/bibtex/)

### Others

* [Analysis of Disfluency in Children’s Speech.](https://isca-speech.org/archive/Interspeech_2020/pdfs/3037.pdf) Tran *et al.* INTERSPEECH 2020. [[bib]](https://isca-speech.org/archive/Interspeech_2020/abstracts/3037.html)

* [Speech disfluencies occur at higher perplexities.](https://www.aclweb.org/anthology/2020.cogalex-1.11.pdf) Sen. Cognitive Aspects of the Lexicon Workshop 2020. [[bib]](https://www.aclweb.org/anthology/2020.cogalex-1.11.bib)

* [Controllable time-delay transformer for real-time punctuation prediction and disfluency detection.](https://arxiv.org/pdf/2003.01309.pdf) Chen *et al.* ICASSP 2020. [[bib]](https://dblp.org/rec/conf/icassp/ChenCLW20.html?view=bibtex)

* [Expectation and locality effects in the prediction of disfluent fillers and repairs in English speech.](https://www.aclweb.org/anthology/N19-3015.pdf) Dammalapati *et al.* NAACL Student Research Workshop 2019. [[bib]](https://www.aclweb.org/anthology/N19-3015.bib)

* [Disfluencies and human speech transcription errors.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3134.pdf) Zayats *et al.* INTERSPEECH 2019. [[bib]](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/3134.html) [[data]](https://github.com/vickyzayats/switchboard_corrected_reannotated)

* [Unediting: detecting disfluencies without careful transcripts.](https://www.aclweb.org/anthology/N15-1161.pdf) Zayats *et al.* NAACL 2015. [[bib]](https://www.aclweb.org/anthology/N15-1161.bib)

* [The role of disfluencies in topic classification of human-human conversations.](http://www.icsi.berkeley.edu/pubs/speech/roleofdisfluencies05.pdf) Boulis *et al.* AAAI Workshop 2005.

# Theses

* [Preliminaries to a theory of speech disfluencies.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.26.1977&rep=rep1&type=pdf) Shriberg. PhD Thesis 1994. [[bib]](https://www.bibsonomy.org/bibtex/164e23b6c85366875aae36bf4133b113e/nlp)


# Contact
Paria Jamshid Lou <paria.jamshid-lou@hdr.mq.edu.au>
