# Of Interest

This readme doc contains links and literature of general interest to the DSML team. It can range from coding blogs, stat information, or the latest machine learning papers of interest. 

Please get in touch if you have suggestions about materials you would like covered.

Thank you, [Andy Wheeler, PhD](mailto:andrew.wheeler@hms.com)

------

## Updated Academic papers 11/25/2020

 - From Sanjeev
   - Google, [Attention is all you need](https://arxiv.org/abs/1706.03762)
   - OpenAI, [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

 - From Andy
   - Variational auto-encoders
     - Original paper, [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) 
     - That is a tough read, I think this blog post is easier! [*Variational auto-encoder in pytorch*](https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/)
   - [*An Explainable Attention Network for Fraud Detection in Claims Management*](http://www.farbmacher.de/working_papers/Farbmacher_etal_2020.pdf)

 - From Jingjie
   - Google, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
   - OpenAI, [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
   - Amazon Alexa, [Optimal Subarchitecture extraction for BERT](https://arxiv.org/pdf/2010.10499.pdf)
   - [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf)

---------

Of note, after here was the old version in chronological order. I think going forward we should put the new stuff at the top though, so older stuff is at the bottom.

## Weekly Links 12/23/2019

 - Chris Stuchio's blog is really good
   - Need to build sanity checks into production models to make sure a simple rule like [99th percentile doesn't misbehave](https://www.chrisstucchio.com/blog/2019/sql_queries_are_data_science_models.html)
   - [Don't use Hadoop - your data isn't that big](https://www.chrisstucchio.com/blog/2013/hadoop_hatred.html)
   - [Example Bayesian modelling using pymc](https://www.chrisstucchio.com/blog/2013/hadoop_hatred.html)

 - [How to analyse 100 GB of data on your laptop with Python](https://towardsdatascience.com/how-to-analyse-100s-of-gbs-of-data-on-your-laptop-with-python-f83363dda94)
    - Uses a python library, `vaex`, to do exploratory data analysis on hdf5 dataset 
 
 - [Analyzing large data on your laptop with a database and R](http://freerangestats.info/blog/2019/12/22/nyc-taxis-sql)
  - very similar to the vaex blog post, but simply uses a MySQL database to pre-aggregate the data

 - AI for Healthcare conference 
   - [blog post talking about the conference](https://towardsdatascience.com/ai-for-healthcare-c975ffad1e8b)
   - [Conference website](https://ai4.io/healthcare/)

 - Interpretable Machine learning
   - [Python blog post](https://towardsdatascience.com/an-overview-of-model-explainability-in-modern-machine-learning-fc0f22c8c29a)
   - [Free online book by Molnar](https://christophm.github.io/interpretable-ml-book/). 
      - He uses R, but many have/can easily be implemented in Python (and often has references to python packages)

# Weekly Links 12/30/2019

 - StatQuest Youtube videos (so can't watch at work)
   - [XGBoost explanation is really good](https://www.youtube.com/watch?v=OtD8wVaFm6E)
   - Other videos on SVM was good as well

 - NeurIps Machine Learning Conference
   - [What we learned from neurips 2019 data](https://medium.com/@NeurIPSConf/what-we-learned-from-neurips-2019-data-111ab996462c)
   - [Key trends in Neurips 2019](https://huyenchip.com/2019/12/18/key-trends-neurips-2019.html)
   - Is this applicable for our work? Seem to have a practitioner tract

# Weekly Links 1/6/2020

 - Using [multiprocessing in Python](https://medium.com/@urban_institute/using-multiprocessing-to-make-python-code-faster-23ea5ef996ba)
 - [AI in healthcare and risks](https://www.scientificamerican.com/article/artificial-intelligence-is-rushing-into-patient-care-and-could-raise-risks/)
 - [Clearing the confusion: Unbalanced Class Data](https://github.com/matloff/regtools/blob/master/UnbalancedClasses.md)
    - Has a formula to adjust predicted probabilities if the proportion changes over time
 - [Panel dashboards for PyData](https://www.youtube.com/watch?v=AXpjbJUVeb4) 
    - Very cool and simple to make interactivity. Do we have access to a Python server?

# Weekly Links 1/13/2020

 - Data drift & encoding categorical variables from Gael Varoquaux work
   - [Comparing distributions use l1 distance](http://gael-varoquaux.info/science/comparing-distributions-kernels-estimate-good-representations-l1-distances-give-good-tests.html)
   - [dirty_cat python package](https://github.com/dirty-cat/dirty_cat/) for working with high cardinality categorical variables (very relevant for diagnoses codes)
   - publications [of results](https://project.inria.fr/dirtydata/publications/)

 - New York state has many [example health datasets](https://healthdata.ny.gov/) free and publicly available that we may use for illustrations.

 - [Multi-armed bandit problem](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)
   - I think this is applicable for many of our projects claim selections, which we only get particular feedback
   - A longer overview chapter on [Multi-armed bandits](https://arxiv.org/pdf/1904.07272.pdf)
   - The book *[Algorithms to Live by](https://www.amazon.com/Algorithms-Live-Computer-Science-Decisions/dp/1627790365)* has a chapter on them as well, for historical overview.

 - Via Bo, DFW Data Science Meetup  (Need to get link at home, Meetup blocked on network computers)

# Weekly Links 1/20/2020

 - Numpy Resources
   - [Minimizing copying data in python to save memory](https://pythonspeed.com/articles/minimizing-copying/)
   - [Dive into Deep Learning (via Numpy) EBook](https://d2l.ai/index.html)
   - [From Python to Numpy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) online book
   - [Numpy Resources](https://brohrer.github.io/numpy_resources.html)
 - SQL Stuff
   - [Top 10 SQL Serve Management Studio Tips and Tricks](https://www.sqlmatters.com/Articles/Top%2010%20SQL%20Server%20Management%20Studio%20(SSMS)%20Tips%20and%20Tricks.aspx) This is not about SQL code, but nice settings to use in the IDE.
   - [12 Common Mistakes and Missed Optimization in SQL](https://hakibenita.com/sql-dos-and-donts) This is about executing SQL code.
 - Other Links
   - [Interactive ROC Visualizer](http://www.navan.name/roc/) via Imad
   - Check out the Panel Dashboards in 1/6/2020 week above. Can do something like this and have confusion table + natural frequency tree (based on prevalence in sample)
   - [Adaptive Swarm Balancing Algorithms for rare-event prediction in imbalanced healthcare data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180830) (via Sanjeev)

# Weekly Links 1/27/2020

 - Stats
   - [Statistical Methods for Machine Learning](http://www.stat.cmu.edu/~larry/=sml/) Larry Wasserman course notes
   - [Overly Optimistic Prediction Results on Imbalanced Data: Flaws and Benefits of Applying Over-sampling](https://arxiv.org/abs/2001.06296) arXiv paper using examples predicting preterm birth
     - plus [GitHub code](https://github.com/GillesVandewiele/EHG-oversampling/) showing how to use smote correctly
   - [Adversarial attacks on medical machine learning](https://science.sciencemag.org/content/363/6433/1287) short Science column (billing makes more sense than the imaging example)
   - [Explainable machine-learning predictions for the prevention of hypoxaemia during surgery](https://www.nature.com/articles/s41551-018-0304-0)


 - Python
   - [Things you’re probably not using in Python 3 – but should](https://datawhatnow.com/things-you-are-probably-not-using-in-python-3-but-should/)
     - simple memoization decorator, pathlib, formatting strings
   - [Intro to Functional programming in Python](https://julien.danjou.info/python-and-functional-programming/) and [Advanced Functional Programming in Python](https://julien.danjou.info/python-functional-programming-lambda/)
   - [RISE](https://rise.readthedocs.io/en/maint-5.6/) turns Jupyter notebook into a reveal.js presentation deck (also has PDF export)
   - [datatable package](https://www.kaggle.com/sudalairajkumar/getting-started-with-python-datatable/notebook) for python. Faster read and manipulation for large datasets than pandas.
   - [Automated Hyperparameter Optimization](https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb)

# Weekly Links 2/03/2020

 - Stats
   - [Time Series Analysis](https://www.economodel.com/time-series-analysis) course notes
     - Also see Rob Hyndman's book, [*Forecasting: Principles and Practice*](https://otexts.com/fpp2/) as well.
   - [Numerical Tours in Python](http://www.numerical-tours.com/python/), various machine learning, image processing, and optimization examples
   - [Paralellizing Monte Carlo Simulation in Python](https://wiseodd.github.io/techblog/2016/06/13/parallel-monte-carlo/)
   - [Anatomy of a Logistic Growth Curve](https://www.tjmahr.com/anatomy-of-a-logistic-growth-curve/) R code, but very nice example of annotating a chart with text/equations
   - [Natural Language Processing Book](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
   - [Frank Harrell suggests you need 20,000+ cases to effectively use split-sample validation](https://www.fharrell.com/post/split-val/). Otherwise should use bootstrapping.

 - Python
   - [Books with Jupyter](https://jupyterbook.org/intro.html)
   - [Dask](https://dask.org/) for working with large datasets

# Weekly links 2/17/2020

 - Python
   - [Overview of itertools](https://www.blog.pythonlibrary.org/2016/04/20/python-201-an-intro-to-itertools/)
   - [Deep reinforcement learning for supply chain and price optimization](https://blog.griddynamics.com/deep-reinforcement-learning-for-supply-chain-and-price-optimization/)
     - includes several Jupyter notebooks [illustrating the techniques](https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/supply-chain/supply-chain-reinforcement-learning.ipynb)

 - Stats
   - [A/B testing and optimizing lift instead of power](https://chris-said.io/2020/01/10/optimizing-sample-sizes-in-ab-testing-part-I/), includes python code.

# Weekly links 2/24/2020

 - Machine Learning
   - [Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)
   - [Are categorical variables getting lost in your random forests?](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/) (H20 can use categorical variables in decision trees)
   - [H2O Python module](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html)

# Weekly links 3/2/2020

 - Machine Learning
   - [Slides on tuning random forests](https://github.com/WhyR2019/presentations/blob/master/Keynotes/Marvin_Wright_RF.pdf)
     - number of variables selected for many columns is more important, outcome ordering of high categories, number of trees for high dim
   - [Dynamic Kernal Matching for sequences](https://github.com/jostmey/dkm)
     - relevant for diag codes as sequences
   - [Jeremy Howard's fastai book](https://github.com/fastai/fastbook) on deep learning as a series of jupyter notebooks on github.

 - Programming
   - [Auto-syncing a git repository](https://jakemccrary.com/blog/2020/02/25/auto-syncing-a-git-repository/) 

# Weekly links 3/9/2020

 - Machine Learning
   - [https://brocktibert.com/post/tensorflow-tip-ins-and-tableau-oh-my/](https://brocktibert.com/post/tensorflow-tip-ins-and-tableau-oh-my/)
     - notebook on using R and python to make a set of embeddings

# Weekly links 3/16/2020

 - Statistics
   - Cam Davidson-Pilon's blog is a very good resource on survival analysis (creator of python `lifelines` and `lifetimes` packages). 
     - [SaaS churn and piecewise regression survival models](https://dataorigami.net/blogs/napkin-folding/churn)
     - [Lifetimes: Measuring Customer Lifetime Value in Python](https://dataorigami.net/blogs/napkin-folding/18868411-lifetimes-measuring-customer-lifetime-value-in-python)
     - [Piecewise exponential models and creating custom models (Survival)](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Piecewise%20Exponential%20Models%20and%20Creating%20Custom%20Models.html)
     - [Time-lagged conversion rates and cure models](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Modelling%20time-lagged%20conversion%20rates.html)

# Weekly links 4/15/2020

 - Python
   - [Text Similarities : Estimate the degree of similarity between two texts](https://medium.com/@adriensieg/text-similarities-da019229c894)
     - various examples of using pre-built libraries and custom functions
   - [Pandas 100 Tricks](https://www.kaggle.com/python10pm/pandas-100-tricks)

 - Unix
   - [typesetting-markdown-part-1](https://dave.autonoma.ca/blog/2019/05/22/typesetting-markdown-part-1/)
     - not so much for the markdown parsing (can be good for docs), but for the very detailed writeup of how to make a nice bash script with arguments