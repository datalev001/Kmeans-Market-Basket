import pandas as pd
import os
import numpy as np


##########data clean and processing
# Read the data from 'online_retail_II.xlsx' file
tran_df = pd.read_excel('online_retail_II.xlsx')
# Define conditions for data cleaning
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity'] > 0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
# Apply data cleaning conditions
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]
# Define columns for duplicate removal
grp = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate']
# Remove duplicated rows based on defined columns
tran_df = tran_df.drop_duplicates(grp)
# Create a new column 'transaction_date' with the date portion of 'InvoiceDate'
tran_df['transaction_date'] = tran_df['InvoiceDate'].dt.date
# Count the occurrences of each product description
cats = tran_df['Description'].value_counts().reset_index()
# Select product descriptions with a count of more than 600 occurrences
cats_tops = cats[cats.Description > 200]
# Create a list of selected product descriptions
pro_lst = list(set(cats_tops['index']))
# Filter the dataset to include only the selected product descriptions
tran_df_sels = tran_df[tran_df['Description'].isin(pro_lst)]


########## word embedding and k-means segmentation
def get_embedding(text):
   result = openai.Embedding.create(input=text, model="text-embedding-ada-002")
   result_text = np.array(result['data'][0]['embedding'])
   return result_text

cats_tops_clus = cats_tops_des.copy()
cats_tops_clus['ada_embedding'] = cats_tops_clus['description'].apply(get_embedding)

word_array = np.array(cats_tops_clus['ada_embedding'].to_list())
word_df = pd.DataFrame(word_array, columns=[f'v_{i}' for i in range(word_array.shape[1])])
cats_tops_clus = pd.concat([cats_tops_clus['description'], word_df], axis=1)

k = 15 
kmeans = KMeans(n_clusters=k)
cats_tops_clus['cluster'] = kmeans.fit_predict(word_df)
cats_tops_clus = cats_tops_clus[['description', 'cluster']]
product_desc_clus_cnt = cats_tops_clus['cluster'].value_counts().reset_index()
product_desc_clus_cnt.columns = ['segment_name', 'count']

##########
tran_df_sels_agg = tran_df_sels.groupby\
    (['Customer ID', 'product'])['Quantity'].sum().reset_index()
   tran_df_sum_pv = tran_df_sels_agg.pivot(index='Customer ID',\
        columns='product', values='Quantity').reset_index()

tran_df_sum_pv = tran_df_sum_pv.fillna(0.0)


######### Customer-Product Transaction Matrix and transformation
def transform_to_trans(transaction, ID, products_pattern):
    # Get the list of product columns (excluding 'customer_id') where values are not 0
    product_columns = [col for col in \
    transaction.columns if col != 'Customer ID' \
    and any(transaction[col] != 0)]
    # Create the 'products' column as a list of product names for each row
    transaction[products_pattern] = transaction.apply(
        lambda row: [col for col in product_columns if row[col] != 0],
        axis=1
    )
   # Create the 'trans' DataFrame with 'customer_id' and 'products' columns
    trans = transaction[[ID, products_pattern]]
    return trans
transaction = transform_to_trans(tran_df_sum_pv[cols], 'Customer ID', 'products_pattern')


######filter the data frame of 'Market Basket Rules' to retain only 'item sets' with two elements 
# The function returns the top N support rules obtained from Market Basket analysis
def filter_and_select_top(D, N):
    # Filter the DataFrame to keep only itemsets with 2 elements
    D['items'] = D['itemsets'].apply(list)
    filtered_D = D[D['items'].apply(len) == 2]
    # Sort the filtered DataFrame by support in descending order and select the top 5 records
    sorted_filtered_D = filtered_D.sort_values(by='support', ascending=False).head(N)
    # Create the final DataFrame DD with only 'support' and 'items' columns
    DF = sorted_filtered_D[['support', 'items']]
return DF


# The function to create 'Market Basket Rules' based centroid, which is a list of data frames
def get_cent(tran_df_sels_agg, cus_data, clus_n):
    tran_df_df = pd.merge(tran_df_sels_agg, cus_data, on = ['Customer ID'], how = 'inner')
    res_lst = []
    for it in range(1, clus_n + 1):
        data_tr = tran_df_df[tran_df_df.seg == it]
        basket_matrix = (data_tr.groupby([transaction_key, item_key])[qty]\
                  .sum().unstack().reset_index().fillna(0).set_index(transaction_key))
        res_basket = basket_analysis(basket_matrix)
        res_basket1 = res_basket[1]    
        res = filter_and_select_top(res_basket1, top_chains)
        res_lst.append(res)
    return res_lst
centroids = get_cent(tran_df_sels_agg,cus_df, clus_n)

########define mew distance used in the kmeans method
def distance(centroids, products_LST): 
    dis_lst = []
    for it1 in products_LST:
        b = []
        for it2 in centroids:
            dis = dist(it2, it1)
            b.append(dis)
        dis_lst.append(b)       
    return dis_lst

# use the function to calculate distance
dis_lst = distance(centroids, products_LST)


#####apply kmeans method to greate market bascket segments
def clus(cus_df, clus_n, max_it):
    # Select relevant columns from the DataFrame based on provided variable names (v)
    selected_columns = ['Customer ID']
    DF = cus_df[selected_columns]
    # Initialize centroids with random data points
    centroids = get_cent(tran_df_sels_agg, cus_df, clus_n)
    RES_DF_old = DF[['Customer ID']]
    RES_DF_old['seg_old'] = np.random.randint(1, 7, size=len(RES_DF_old))

    for i in range(max_it):
        # Calculate distances and assign points to clusters
        distances = distance(centroids, products_LST)
        DF['seg'] = np.argmin(distances, axis=1) + 1
        # Update centroids
        new_centroids = get_cent(tran_df_sels_agg, DF, clus_n)
        centroids = new_centroids[:]
        print('iteration time: ', str(i))
        RES_DF_new = DF[['Customer ID', 'seg']]
        RES_DF_new_old = pd.merge(RES_DF_new, RES_DF_old, on=['Customer ID'], how='inner')
        seg = RES_DF_new_old.seg
        seg_old = RES_DF_new_old.seg_old
        crs = pd.crosstab(seg, seg_old).reset_index()
        crs = crs.drop(['seg'], axis=1)
        nms = list(crs.columns)
        crs.columns = ['seg_' + str(it) for it in nms]
        crs['error'] = 1 - crs[list(crs.columns)].max(axis=1) / crs[list(crs.columns)].sum(axis=1)
        error = crs['error'].mean()
        print('error: ', str(error))
        RES_DF_old = RES_DF_new.copy()
        RES_DF_old.columns = ['Customer ID', 'seg_old']
        # Check for convergence
        if error < 0.01:
            break
    # Create the RES_DF DataFrame with cus_id and cluster columns
    RES_DF = DF[['Customer ID', 'seg']]

    return RES_DF

# Test the clus function
max_it = 15
result_df = clus(cus_df, clus_n, max_it)
