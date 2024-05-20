# 推荐系统
推荐系统从入门到精通，本项目全面介绍了工业级推荐系统的理论知识（王树森推荐系统公开课-基于小红书的场景讲解工业界真实的推荐系统），如何基于TensorFlow2训练模型，如何实现高性能、高并发、高可用的Golang推理微服务。以及一些Sklean和TensorFlow编程基础知识。Comprehensively introduced the theory of industrial recommender system base on deep learning, how to trainning models based on TensorFlow2, how to implement the high-performance、high-concurrency and high-available inference services base on Golang.  
注意：第一部分的理论知识在本仓库，第二、三、四部分的代码在其他仓库，点击链接即可跳转。


## 注意
如果Github打开jupyter notebook发生错误，可以点击“Backup Link”通过 https://nbviewer.org 间接访问单个链接。  
或者通过以下链接访问整个项目：  
●  [Recommender_System](https://nbviewer.org/github/solidglue/Recommender_System/tree/master/)  


## 一、推荐系统理论 (Recommender System Theory)
王树森推荐系统公开课 - 基于小红书的场景讲解工业界真实的推荐系统，读书笔记。

### 01 概要 (Introduce)
●  [推荐系统的链路](https://github.com/solidglue/Recommender_System/blob/master/01_Basic/01_01_Recommend_flow.ipynb)        [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/01_Basic/01_01_Recommend_flow.ipynb)]  
●  [AB测试](https://github.com/solidglue/Recommender_System/blob/master/01_Basic/01_02_AB_test.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/01_Basic/01_02_AB_test.ipynb)  

### 02 召回 (Recall)
●  [基于物品的协同过滤（ItemCF）](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_01_Item_cf.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_01_Item_cf.ipynb)  
●  [Swing召回通道](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_02_Swing.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_02_Swing.ipynb)  
●  [基于用户的协同过滤（UserCF）](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_03_User_cf.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_03_User_cf.ipynb)  
●  [离散特征处理](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_04_Discrete_feature.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_04_Discrete_feature.ipynb)  
●  [矩阵补充](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_05_Matrix_completion.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_05_Matrix_completion.ipynb)  
●  [双塔模型：模型和训练](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_06_Twotower_model_and_training.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_06_Twotower_model_and_training.ipynb)  
●  [双塔模型：正负样本](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_07_Twotower_positive_and%20negtive_samples.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_07_Twotower_positive_and%20negtive_samples.ipynb)  
●  [双塔模型：线上召回和更新](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_08_Twotower_serving.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_08_Twotower_serving.ipynb)  
●  [双塔模型+自监督学习](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_09_Twotower_and_selfupervised_learning.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_09_Twotower_and_selfupervised_learning.ipynb)  
●  [Deep Retrieval召回](https://github.com/solidglue/Recommender_System/blob/master/02_Recall/02_10_Deep_retrieval.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_10_Deep_retrieval.ipynb)  
●  [其他召回通道](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_11_Geo_author_cache_recall.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_11_Geo_author_cache_recall.ipynb)    
●  [曝光过滤和Bloom Filter](https://github.com/solidglue/recommender_system/blob/master/02_Recall/02_12_Exposure_and_bloom_filter.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/02_Recall/02_12_Exposure_and_bloom_filter.ipynb)  

### 03 排序 (Ranking)
●  [多目标排序模型](https://github.com/solidglue/Recommender_System/blob/master/03_Rank/03_01_Multi_task_model.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/03_Rank/03_01_Multi_task_model.ipynb)    
●  [MMoE](https://github.com/solidglue/Recommender_System/blob/master/03_Rank/03_02_mmoe.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/03_Rank/03_02_mmoe.ipynb)    
●  [预估分数融合](https://github.com/solidglue/Recommender_System/blob/master/03_Rank/03_03_Weight_score.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/03_Rank/03_03_Weight_score.ipynb)  
●  [视频播放建模](https://github.com/solidglue/Recommender_System/blob/master/03_Rank/03_04_Video_model.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/03_Rank/03_04_Video_model.ipynb)  
●  [排序模型的特征](https://github.com/solidglue/Recommender_System/blob/master/03_Rank/03_05_Ranking_model_features.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/03_Rank/03_05_Ranking_model_features.ipynb)  
●  [粗排模型](https://github.com/solidglue/Recommender_System/blob/master/03_Rank/03_06_Preranking.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/03_Rank/03_06_Preranking.ipynb)  

### 04 特征交叉 (Feature Cross)
●  [因子分解机FM](https://github.com/solidglue/Recommender_System/blob/master/04_Cross/04_01_FM.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/04_Cross/04_01_FM.ipynb)  
●  [深度交叉网络DCN](https://github.com/solidglue/Recommender_System/blob/master/04_Cross/04_02_DCN.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/04_Cross/04_02_DCN.ipynb)   
●  [LHUC网络结构](https://github.com/solidglue/Recommender_System/blob/master/04_Cross/04_03_LHUC.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/04_Cross/04_03_LHUC.ipynb)  
●  [SENet Bilinear Cross](https://github.com/solidglue/Recommender_System/blob/master/04_Cross/04_04_SENet_Bilinear_cross.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/04_Cross/04_04_SENet_Bilinear_cross.ipynb)  

### 05 行为序列 (User Behavior Sequence)
●  [用户行为序列建模](https://github.com/solidglue/Recommender_System/blob/master/05_LastN/05_01_User_behavior_sequence.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/05_LastN/05_01_User_behavior_sequence.ipynb)  
●  [DIN模型（注意力机制）](https://github.com/solidglue/Recommender_System/blob/master/05_LastN/05_02_DIN.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/05_LastN/05_02_DIN.ipynb)  
●  [SIM模型（长序列建模）](https://github.com/solidglue/Recommender_System/blob/master/05_LastN/05_03_SIM.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/05_LastN/05_03_SIM.ipynb)  

### 06 重排 (Re-rank)
●  [物品相似性的度量、提升多样性的方法](https://github.com/solidglue/Recommender_System/blob/master/06_Rerank/06_01_Diversity.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/06_Rerank/06_01_Diversity.ipynb)   
●  [MMR多样性算法](https://github.com/solidglue/Recommender_System/blob/master/06_Rerank/06_02_MMR.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/06_Rerank/06_02_MMR.ipynb)  
●  [业务规则约束下的多样性算法](https://github.com/solidglue/Recommender_System/blob/master/06_Rerank/06_03_Rerank_rules.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/06_Rerank/06_03_Rerank_rules.ipynb)  
●  [DPP多样性算法（上）](https://github.com/solidglue/Recommender_System/blob/master/06_Rerank/06_04_DPP_01.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/06_Rerank/06_04_DPP_01.ipynb)   
●  [DPP多样性算法（下）](https://github.com/solidglue/Recommender_System/blob/master/06_Rerank/06_05_DPP_02.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/06_Rerank/06_05_DPP_02.ipynb)   

### 07 物品冷启动 (Cold Start)
●  [优化目标&评价指标](https://github.com/solidglue/Recommender_System/blob/master/07_Cold_start/07_01_Optimization_objectives_and_evaluation_metrics.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/07_Cold_start/07_01_Optimization_objectives_and_evaluation_metrics.ipynb)   
●  [简单的召回通道](https://github.com/solidglue/Recommender_System/blob/master/07_Cold_start/07_02_Simple_recall.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/07_Cold_start/07_02_Simple_recall.ipynb)  
●  [聚类召回](https://github.com/solidglue/Recommender_System/blob/master/07_Cold_start/07_03_Clustering_recall.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/07_Cold_start/07_03_Clustering_recall.ipynb)  
●  [Look-Alike召回](https://github.com/solidglue/Recommender_System/blob/master/07_Cold_start/07_04_Look_a_like_recall.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/07_Cold_start/07_04_Look_a_like_recall.ipynb)  
●  [流量调控](https://github.com/solidglue/Recommender_System/blob/master/07_Cold_start/07_05_Network_flow_control.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/07_Cold_start/07_05_Network_flow_control.ipynb)  
●  [冷启动的AB测试](https://github.com/solidglue/Recommender_System/blob/master/07_Cold_start/07_06_Cold_start_abtest.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/07_Cold_start/07_06_Cold_start_abtest.ipynb)  

### 08 推荐系统涨指标的方法 (Improvement)
●  [概述](https://github.com/solidglue/Recommender_System/blob/master/08_Improvement/08_01_Improvement_basic.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/08_Improvement/08_01_Improvement_basic.ipynb)  
●  [召回](https://github.com/solidglue/Recommender_System/blob/master/08_Improvement/08_02_Improvement_recall.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/08_Improvement/08_02_Improvement_recall.ipynb)  
●  [排序](https://github.com/solidglue/Recommender_System/blob/master/08_Improvement/08_03_Improvement_rank.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/08_Improvement/08_03_Improvement_rank.ipynb)  
●  [多样性](https://github.com/solidglue/Recommender_System/blob/master/08_Improvement/08_04_Improvement_diversoty.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/08_Improvement/08_04_Improvement_diversoty.ipynb)  
●  [特征用户人群](https://github.com/solidglue/Recommender_System/blob/master/08_Improvement/08_05_Improvement_special_user_group.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/08_Improvement/08_05_Improvement_special_user_group.ipynb)  
●  [交互行为（关注、转发和评论）](https://github.com/solidglue/Recommender_System/blob/master/08_Improvement/08_06_Improvement_interaction_behavior.ipynb)         [         (Backup Link)](https://nbviewer.org/github/solidglue/Recommender_System/blob/master/08_Improvement/08_06_Improvement_interaction_behavior.ipynb)  


## 二、TensorFlow2模型训练 (Model Trainning)
以"DNN_for_YouTube_Recommendations"模型和电影评分数据集（ml-1m）为基础，详尽的展示了如何基于TensorFlow2训练推荐系统排序模型。  
● [YouTube深度排序模型(多值embedding、多目标学习)](https://github.com/solidglue/DNN_for_YouTube_Recommendations)  


## 三、模型推理服务Golang (Inferecnce Services)
基于Goalng、Docker和微服务思想实现了高并发、高性能和高可用的推荐系统推理微服务，包括多种召回/排序服务，并提供多种接口访问方式（REST、gRPC和Dubbo）等，每日可处理上千万次推理请求。  
● [推荐系统推理微服务Golang](https://github.com/solidglue/Recommender_System_Inference_Services)  


## 四、编程基础 (Sklearn/TensorFlow)
●  [机器学习Sklearn入门教程](https://github.com/solidglue/Machine_Learning_Sklearn_Examples)  
●  [深度学习TensorFlow入门教程](https://github.com/solidglue/Deep_Learning_TensorFlow2_Examples)  
