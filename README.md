# SafeSF - Future of Traffic Safety

In the US, 42,000 traffic deaths and millions more injuries occur each year. 31% of the accidents occur in places where no other accidents happened nearby (within 50 meters) within four years.  With prediction machine learning algorithms, our goal is to help reduce traffic accidents within San Francisco by providing a map visualization that warns our users if the road that they plan to take is classified as more dangerous than others.

<img width="1920" height="840" alt="Capture2" src="https://github.com/user-attachments/assets/97b8635e-b01f-4bfa-8478-9dd073c6645d" />

<div align="center">Fig 0. Output of our ML model predicting collisions.</div>

# Data Sources

- [TransBASE Dashboard - City of San Francisco: Traffic Collision](https://data.sfgov.org/Public-Safety/Traffic-Crashes-Resulting-in-Injury/ubvf-ztfx)
- [National Oceanic & Atmospheric Administration: Satellite images](https://coast.noaa.gov/dataviewer/#/imagery/search/-13639308.671664422,4538340.267087154,-13620556.11994757,4552710.429571103)
- [United States Census Bureau: Road types and vectors](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Roads)
- San Francisco Municipal Transportation Agency: [Stop signs](https://data.sfgov.org/Transportation/Stop-Signs/4542-gpa3/about_data), [Bus stops](https://data.sfgov.org/Transportation/Muni-Stops/i28k-bkz6/about_data), and [Paving](https://data.sfgov.org/d/5wbp-dwzt/about) datasets
- [uszipcode python database](https://pypi.org/project/uszipcode/)
- Extracted [OpenStreetMap](https://www.openstreetmap.org/#map=4/38.01/-95.84) data using [PyROSM](https://pyrosm.readthedocs.io/en/latest/) library. Street features from OpenStreetMap were found to have several null values and were not joined in the dataset.

# Machine Learning Process

## Recommended Model
Multinomial Logistic Regression
The Multinomial Logistic Regression model is the recommended model used to predict the multi-class classification bins of traffic collisions. The model was observed to have the highest weighted F1 score and significantly shorter training time compared to the other modeling experiments performed.

<img width="680" height="517" alt="image" src="https://github.com/user-attachments/assets/a4f4a84b-ad30-4e53-a37b-899753797251" />
<div align="center">Fig. A: Multinomial Logistic Regression Model Training Loss Curve</div>
<br>

<img width="602" height="648" alt="image" src="https://github.com/user-attachments/assets/c7f0c0cc-1f3a-4e96-b1af-760f6d5c4869" />
<div align="center">Fig. B: Confusion Matrix of Normalized Class F1 Scores</div>
<br>

<img width="970" height="314" alt="image" src="https://github.com/user-attachments/assets/bc6a6dc8-671a-4617-9958-ddf519daf4f9" />
<div align="center">Fig. C: Multi-Class Classification Model Training Summary</div>
<br>

## Data Pre-Processing
**Split San Francisco map by creating 100m x 100m "tiles"**

<img width="486" height="490" alt="image" src="https://github.com/user-attachments/assets/eaffe3b3-cd1d-4d16-8cdd-03289754fa87" />
<br><br>
Using geographic coordinates, the vertical and horizontal distances from the boundaries of San Francisco were divided by 100 to create the tiles. Each 100m x 100m tile is designated as ID to aggregate features from other data sets.
<br><br>
<img width="1016" height="292" alt="image" src="https://github.com/user-attachments/assets/38493ac5-0f85-4a82-9725-3a723ecb00fc" />
<br>

**Data Join and Pre-processing**

The tiles data frame was expanded with street characteristics from various data sources:
- Collisions data
- Stop signs data
- Street Paving data
- Bus Stops data
- Road Type data
- Satellite Image data
- Zip Codes data

Pre-processing:
- Removed tiles without road data
- Read image files with tifffile package
- Standardized image resolution to 148 x 188
<br>

**Feature Engineering: Splitted Features with historical and future data**

<img width="556" height="342" alt="image" src="https://github.com/user-attachments/assets/7e438545-ace1-43e3-be4e-3c89c46a45bd" />
<br>

The features having dates or timestamps are split into their historical and future features.​ These are:
- Historical Paving and Future Paving
- Historical Number of Collisions and Future Number of Collisions
<br>

**Feature Engineering: Created bins for multi-classification**

<img width="566" height="260" alt="image" src="https://github.com/user-attachments/assets/e26bca54-0f2e-4adc-8b7f-2793eb6599dc" />
<br>

Multi-classification bins were created as the label feature for each "tile" data row. The bins are based on clustering future number of collisions such as "0" for zero future number of collisions, "1" for one or two future number of collisions, "2" for three or four future number of collisions, and so on.
<br>

**Feature Engineering: Created one hot encoded zip codes feature**

<img width="1200" height="348" alt="image" src="https://github.com/user-attachments/assets/25aaded4-8c45-4744-8b2e-6399ffd4685e" />
<br>

Using uszipcode Python database, the zip codes belonging to each 100m x 100m tile are collected and represented in binary vectors through one-hot encoding.
<br>

## Machine Learning Process
**Train - Test - Validation Set Split**

<img width="1152" height="627" alt="image" src="https://github.com/user-attachments/assets/778b702c-8e0b-4c3b-ac88-fdcf961342a3" />
<br>

The dataset is split into 80% training set and 20% test set. Then, 20% of the training set was used as the validation set.<br>

**Class Weights**

To deal with the imbalanced training data as shown in the histogram below, a balancing class weights parameter was implemented in the models with the aid of sklearn's compute_class_weight function to estimate the class weights.
<br>

<img width="839" height="451" alt="image" src="https://github.com/user-attachments/assets/6240b081-e292-4546-ad0b-a8d1f37e365a" />
<div align="center">Fig. A: Histogram of Multi-Class Bins in Train Set</div><br>

<img width="480" height="542" alt="image" src="https://github.com/user-attachments/assets/f97e5d31-6b6e-429b-a4de-74454cd8628f" />
<div align="center">Fig. B: Dictionary of Class Weights Used in Modeling</div><br>

**Modeling**

The team focused on modeling using multi-class classification task to predict the traffic collision (represented by the bins) in San Francisco. 

Features used as inputs (x-variables):
- Stop Signs
- Bus stops
- Road types
- Historical number of collisions
- Historical paving projects
- Satellite images
- Zip codes

Features used as output (y-variable): Classification Bins of Future Collisions​

Models explored:

- Multi-class classification model

The Multinomial Logistic Regression model was used as the baseline model during the machine learning experimentation. Since the Multinomial Logistic Regression model performed the best compared to all models explored in terms of F1 score and training time, our team selected this model as the final model. The final results are summarized in conclusion section below.

- Random Forest and Gradient Boosted Tree models

Failed experiments due to training training errors caused by the big array size input of (2095, 111348) with full colors or (8376, 55700) with greyscaled colors. Greyscaling is done by averaging the red, green and blue colors, reducing the array size by 50%.

- Convolutional Neural Network (CNN) model

Below is a summary of CNN experiments​ performed. Among the CNN "large" experiments, it was found that training with ResNet-18 architecture and balanced class weights produced the best results. CNN "small" experiments were also conducted by replacing the ResNet architecture with CNN and global pooling layers to significantly reduce the number of parameters from 27 million to 26 thousand. Reducing the amount of parameters in CNN "small" experiments bring them closer Multinomial Logistic Regression model's number of parameters for comparability. The results of CNN "small" experiments outperformed those of CNN "large" experiments.

<img width="956" height="380" alt="image" src="https://github.com/user-attachments/assets/e2f1e135-2f5d-4f99-b393-ee8633919eba" />
<div align="center">Fig. A: Convolutional Neural Network (Large) Experiments</div><br>

<img width="966" height="380" alt="image" src="https://github.com/user-attachments/assets/bb7b8447-507a-42f5-968d-1ea11de9f01f" />
<div align="center">Fig. B: Convolutional Neural Network (Small) Experiments</div><br>

## Conclusion
The **Multinomial Logistic Regression model** outperformed all of the Convolutional Neural Network (CNN) model experiments in terms of weighted F1 score and training time. While the weighted score of the Multinomial Logistic Regression model ties with the best CNN "small" model for having the highest weighted F1 score among all experiments, the training time is significantly shorter in Multinomial Logistic Regression model. Hence, the Multinomial Logistic Regression model is selected as the recommended model for predicting the multi-class classification bins of traffic collisions with street characteristics and big image data as inputs.

<img width="975" height="196" alt="image" src="https://github.com/user-attachments/assets/e2a144a6-97b1-48fc-b2e3-4b98ab7ac420" />

Over the course of the project, the team discovered zero shot learning techniques, which may be a better choice. This will be explored in the next phase of the project.

[<img width="25" height="25" alt="image" src="https://github.com/user-attachments/assets/7d7d677f-fa60-490d-ac6c-30532de3bf17" />Check out our GitHub](https://github.com/arisanguyen/210_Capstone_Aditya_Arisa_Noriel)
