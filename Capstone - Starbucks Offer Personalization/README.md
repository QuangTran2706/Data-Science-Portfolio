# Starbucks Offer Personalization 
### Sending the right offer to right people

## Introduction
Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.
Not all users receive the same offer, and that is the challenge to solve with this data set.

## Data 
The data sets contain simulated data that mimics customer behavior on the Starbucks rewards mobile app. It is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Three datasets are provided by Starbucks for this project:
* User profile data: contains users' age, gender, income, membership start date.
* Offer portfolio data: contains information about the offers, including offer type(Buy one get one free/ discount/informational), offer duration, rewards, minimal transaction amount to use the offer etc.
* User transaction log: contains users' transaction amount, usage of offers, time of transaction etc.

## Example
To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

## Challenge
Given the rules and exceptions for offer completion as illustrated in the example, the data cleaning and feature engineering part becomes especially important and tricky.

The key part of the project is summarizing each customer's transaction behaviors from the chronological activity log to attribute the right transactions to the right offer types and track down their conversions for different offers on the user level. Those engineered features will then be combined with the user profile information such as income, age, gender and when their membership started to provide a wholistic view of each customer.

## Goal 
My goal for this project is to use those demographic and behavioral features to predict what the best personalized offer is for each individual customers that would maximize the conversion rate. Also I want to see whether there are any distinct subgroups within the entire customer base that exhibit certain patterns of demographics and buying behaviors, which set them apart from one another.

## Road Map
#### 1. Exploratary Analysis:
Understand the three datasets

#### 2. Feature Engineering:
Create attributes from the transaction log on the user level

#### 3. Machine Learning:
Build multioutput classification model to determine the offer types ranked by likelihood of conversion

#### 4. Clustering:
Find subgroups within customers that Starbucks can target differently according to each group's distinct demographical or behavioral patterns.
