# Kmeans-Market-Basket
Development of a Hierarchical K-Means Clustering Recommender System Informed by Market Basket Analysis

Leveraging Word2Vec and Market Basket Segmentation to Enhance Personalized Recommendation Algorithms

Recommendation systems are vital in the e-commerce industry, as they provide tailored product suggestions to customers based on their preferences and behaviors. While traditional recommendation methods like Collaborative Recommendation and Market Basket (Sequence) Analysis have been widely used, they do come with certain limitations.

Collaborative Recommendation primarily centers on customer behavior similarity, but it can become computationally demanding when dealing with extensive product catalogs. Moreover, it often overlooks the relationships and commonalities among products, especially when valuable information like "product descriptions" is available. 

The Market Basket (Sequence) Analysis aids in uncovering relationships among products by computing support and confidence statistics. However, it tends to overlook the similarities in customer behavior, often leading to generic recommendations that may not align with individual customer preferences. Additionally, the process can generate an overwhelming number of 'Market Basket Rules,' making it challenging for marketers to make decisions.

In this post, we introduce an innovative solution to address the drawbacks of traditional recommendation systems. We present a Hierarchical Comprehensive Recommender System that combines advanced techniques to provide more personalized and effective product recommendations. The core elements of this solution include the follwing steps:

Word2Vec for Production Descriptions:
We begin by harnessing the potent Word2Vec word embedding technique to convert textual product descriptions into high-dimensional vector representations. This transformation enables us to capture the semantic meanings and relationships among distinct products, even when their descriptions are intricate and unique.

Product Segmentation:
Leveraging the vectorized production descriptions, we perform product segmentation, grouping products with akin descriptions. This step augments our comprehension of inherent relationships and similarities among products, facilitating the creation of meaningful product segments.

Market Basket (Sequence) Based K-Means Clustering:
We employ a K-Means Clustering algorithm developed in-house, which utilizes the Market Basket (Sequence) Analysis results. The basket rules and support values guide each iterative K-Means step, ensuring the identification of strong associations. This method results in meaningful, accurate, and relatively balanced segment sizes, unlike the skewed segment sizes found in traditional K-Means approaches.

Following the segmentation of products and customers, these segments serve as the foundation for the Hierarchical recommender system, as outlined below:
  •	Establishment of Recommendation Rules among Product Segments (First Layer)
  •	Establishment of Recommendation Rules within Product Segments (Second Layer)

