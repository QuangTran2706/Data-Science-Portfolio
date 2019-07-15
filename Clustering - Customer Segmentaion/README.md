# Unsupervised Learning - Customer Segmentation
### Project Overview

In this project, I applied unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. 

First KMeans clustering is fit on the PCA transformed general population dataset, then it is used to segment the mail company customer data with the same number of clusters.
I then compare the proportion of data points in each cluster between the general population and the customer data to identify the "overrepresented" clusters 
where there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid).
That suggests the people in that cluster to be a target audience for the company.

These segments of the customer base can be the smaller but more precise audience to target for direct marketing campaigns who will have higher probabilities to convert.


The data for this project is real life demographical and psychographical data provided by a German consulting firm Bertelsmann Arvato Analytics. It consists two individual datasets of the same structure - one for the German general population and the other for customer of the mail-order company(unfortunately too big to be uploaded). There is also a data dictionary with explanations of each column in the data.  

