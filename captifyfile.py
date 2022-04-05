#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pyspark


# In[2]:


import findspark
findspark.init('/home/ola/spark-3.0.3')


# In[3]:


import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
import urllib
import pandas as pd
import wget


# In[4]:


spark = SparkSession.builder.appName ("oladata").getOrCreate()


# In[5]:


import tarfile


# In[6]:


df= urllib.request.urlretrieve("https://cdn.captifytechnologies.com/samples/sampled_kws.tar.gz", "sampled_kws.tar.gz")


# In[7]:


tf = tarfile.open("sampled_kws.tar.gz")


# In[8]:


tf.extractall('./zip_file')


# In[9]:


df1 =spark.read.format("parquet").load("./zip_file/sampled_kws/day=20220222")


# In[10]:


df2 = spark.read.format("parquet").load("./zip_file/sampled_kws/day=20220223")


# In[11]:


df3 = spark.read.format("parquet").load("./zip_file/sampled_kws/day=20220224")


# In[12]:


from functools import reduce
from pyspark.sql import DataFrame


# In[13]:


df_total =[df1,df2,df3]


# In[14]:


#merging all data df1, df2 and df3 together because they
union_df = reduce(DataFrame.unionAll,df_total)


# In[15]:


union_df.show()


# In[16]:


#checking dataframe schema
union_df.printSchema()


# In[17]:


from pyspark.sql import functions as F


# In[18]:


union_df=union_df.withColumn("eventDate", F.col("eventDate").cast("timestamp"))


# In[19]:


union_df.printSchema()


# In[20]:


publisherid_count = union_df.groupBy('publisherid').count()


# In[21]:


publisherid_count_for_graph = publisherid_count.filter(publisherid_count['count'] >1000).sort('count', ascending=False).toPandas()


# In[22]:


list(publisherid_count_for_graph[0:20]['publisherid'])


# In[23]:


from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# In[24]:


#plotting graph to show top publishers

counts = publisherid_count_for_graph[0:20]
counts['publisherid'] = [x.title() for x in counts['publisherid']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y=counts['publisherid'], 
       width=counts['count'])
ax.set_title("Top publishers")
ax.set_xlabel("number_keyword field")
plt.xticks(rotation=90)
ax.invert_yaxis()


# In[25]:


from pyspark.sql.functions import to_date


# In[26]:


first_ten_date = union_df.select(to_date(union_df.eventDate, 'dd-MM-yyyy HH:mm:ss').alias('date'))     .orderBy('eventDate').take(10)


# In[27]:


first_ten_date


# In[28]:


import sys
from operator import add
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
import pyspark.sql.functions as f
from pyspark.ml import Pipeline


# In[29]:


#spliting into keywords and counting
sf=union_df.withColumn('word', f.explode(f.split(f.col('keyphrase'), ' '))).groupBy('word').count().sort('count', ascending=False)


# In[30]:


sf.show()


# In[31]:


#removing null values in dataframe
data_df = union_df.select("publisherId", "keyphrase").filter("keyphrase is Not NULL")


# In[32]:


#show data_df
data_df.show()


# In[33]:


#converts the input string to lowercase and splits it by white spaces
tokenizer = Tokenizer(inputCol="keyphrase", outputCol="words_token")


# In[34]:


#apply transform
tokenized = tokenizer.transform(data_df).select('publisherid','words_token')


# In[35]:


#show tokenized
tokenized.show()


# In[36]:


#filters out stop words from words_token
remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')


# In[37]:


#apply transform
data_clean = remover.transform(tokenized).select('publisherid', 'words_clean')


# In[38]:


data_clean.show()


# In[40]:


result = data_clean.withColumn('word', f.explode(f.col('words_clean')))   .groupBy('word')   .count().sort('count', ascending=False)


# In[41]:


new_result =data_clean.select("*").withColumn('word', f.explode(f.col('words_clean')))   .groupBy('word', 'publisherid')   .count().sort('count', ascending=False)


# In[42]:


result.show()


# In[43]:


new_result.show()


# In[44]:


import pyspark.sql.functions as F


# In[45]:


#create column for length of words
new_result= new_result.withColumn("length_of_word", F.length("word"))


# In[46]:


#show length of words
new_result.show(truncate=False)


# In[47]:


#find when length of word is not zero
new_result=new_result.filter(new_result.length_of_word!=0)


# In[51]:


#show result
new_result.show()


# In[58]:


#dropping length
new_result = new_result.drop("length_of_word")


# In[59]:


new_result.show()


# In[60]:


pandas_df = new_result.toPandas()


# In[ ]:


pandas_df.sort_values(by='count',ascending=False).plot(x ='publisherid', y='count', kind = 'bar')


# In[ ]:


pandas_df.sort_values(by='count',ascending=False).plot(x ='word', y='count', kind = 'bar')


# In[ ]:


list_df = new_result.select("word", "count").rdd.flatMap(lambda x: x).collect()


# In[ ]:


from pyspark.sql.functions import col


# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


# In[ ]:


#select distinct publishers
publisherid_df = union_df.select("publisherId").distinct().rdd.flatMap(lambda x: x).collect()


# In[ ]:




