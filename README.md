# **Predicting Heart Disease: Analysis of Risk Factors with Machine Learning**

## **Introduction**

Welcome to this comprehensive exploratory data analysis (EDA) and predictive modeling project focused on heart disease. In this notebook, we delve deep into understanding the factors influencing heart disease through both univariate and bivariate analysis, unveiling the relationships between various health indicators. Additionally, we aim to build and compare predictive models using three different algorithms: Logistic Regression, RandomForestRegressor, and SVM, to identify which model most accurately predicts the occurrence of heart attacks. This analysis not only sheds light on the underlying patterns and correlations within the dataset but also provides a robust foundation for developing effective preventive measures against heart disease.

## **Dataset Overview**

This dataset contains anonymized health survey data from adults in the United States. The primary objective is to predict heart disease occurrence based on various risk factors and demographic information. The dataset includes the following features:

1.  **Demographic Information**
    -   **Sex**: Gender of the individual (Male/Female).
    -   **AgeCategory**: Age group of the individual (e.g., 18-24, 80 or older).
    -   **Race**: Race or ethnicity of the individual (e.g., White, Black, Hispanic).
2.  **Lifestyle Factors**
    -   **Smoking**: Whether the individual has smoked at least 100 cigarettes in their lifetime
    -   **AlcoholDrinking**: Whether the individual drinks heavily (defined as more than 14 drinks per week for men and more than 7 drinks per week for women).
    -   **PhysicalActivity**: Whether the individual engages in physical activity or exercise outside of their regular job.
    -   **SleepTime**: Average number of hours of sleep per day
3.  **Health Conditions and Diseases**
    -   **HeartDisease**: Whether the individual has been diagnosed with heart disease.
    -   **BMI**: Body Mass Index of the individual.
    -   **Stroke**: Whether the individual has had a stroke.
    -   **PhysicalHealth**: Number of days in the past month when physical health was poor.
    -   **MentalHealth**: Number of days in the past month when mental health was poor.
    -   **DiffWalking**: Whether the individual has difficulty walking or climbing stairs.
    -   **Asthma**: Whether the individual has asthma.
    -   **KidneyDisease**: Whether the individual has kidney disease.
    -   **SkinCancer**: Whether the individual has been diagnosed with skin cancer.
    -   **Diabetic**: Whether the individual has diabetes or pre-diabetes
4.  **General Health Status**
    -   **GenHealth**: Self-reported general health status (Excellent, very good, Good, Fair, Poor)

This dataset can be used to explore correlations between lifestyle factors and various health conditions, especially heart disease. Researchers can identify key risk factors and develop predictive models to improve early detection and preventive measures.

## **Exploratory Data Analysis (EDA) Conclusion**

**Overview of Heart Disease Prevalence**

-   **Prevalence Insight**: Approximately **8.6%** of the individuals surveyed in this dataset are reported to have heart disease, indicating a significant health concern within this population.

    ![1](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/1847c758-7259-4460-acdf-669dad8f8328)

**Analysis of Body Mass Index (BMI)**

-   **Average BMI**: The average BMI among the participants is **28.3**, suggesting a trend towards overweight and obesity across the dataset.
-   **Distribution Characteristics**: The BMI distribution is notably right-skewed, with the majority of the participants possessing a BMI ranging from **20** to **35**, highlighting the prevalence of weight-related health risks.

    ![2](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/1132d244-be37-4c65-91d3-5cb31ff19b7a)

**Smoking and Alcohol Consumption**

-   **Smoking Rates**: A substantial **41.2%** of the dataset's participants are smokers, underscoring a critical public health issue.

    ![3](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/12bc763d-3dd4-422a-86a4-d5a9242ea1f8)

-   **Alcohol Consumption**: **6.8%** of individuals consume alcohol excessively, which poses additional health risks and potential for disease.

    ![4](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/0f10454e-b209-4dc7-89d6-fe8aa8a90b90)

**Stroke Incidence**

-   **Stroke Prevalence**: Strokes have been experienced by **3.8%** of the individuals, linking vascular health issues with other risk factors present in the dataset.

    ![5](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/b60c5c4a-495c-43e5-9141-3062b08ba8c7)

**Physical and Mental Health Metrics**

-   **Physical Health**: On average, participants reported **3.4** days per month of poor physical health, which impacts their overall quality of life.

    ![6](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/b5621474-c550-4778-bf19-cd3035bc6eaf)

-   **Mental Health**: Similarly, the average number of days reported for poor mental health stands at **3.9** days per month, indicating a considerable mental health burden.

    ![7](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/b6455e24-9926-4148-88cb-57c3957c9846)

**Mobility Challenges**

-   **Difficulty Walking**: **13.9%** of the dataset's participants have difficulty walking or climbing stairs, reflecting mobility limitations and potential disability.

    ![8](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/3beb303e-f914-4fdf-83d2-cd598f68239e)

**Demographic Distribution by Sex**

-   **Gender Split**: The dataset is relatively balanced with **52.5%** females and **47.5%** males, allowing for gender-based health analysis.

    ![9](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/763734ac-be84-4492-9e53-15b8a57b7f34)

**Age Category Insights**

-   **Age Variability**: The largest age group within the study is **65-69** years at **11.4%**, followed closely by **60-64** years at **11.2%** and **70-74** years at **10.3%**. The smallest group is those aged **25-29** at **5.7%**, illustrating the dataset's skew towards older adults.

    ![10](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/e3718f4e-95a9-412b-9ed1-a53e35ccd6d6)

**Racial and Ethnic Composition**

-   **Race Distribution**: A majority of **77.1%** identify as White, with Hispanic and Black populations following at **8.6%** and **7.2%** respectively, indicating racial and ethnic variations in the dataset's demographic.

    ![11](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/d1cdc60a-ad8c-4bed-a8df-594e5bd7a496)

**Diabetes Prevalence**

-   **Diabetes Rates**: **13.7%** of participants are living with diabetes, which is critical for understanding broader metabolic health challenges.

    ![12](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/01d9c5dd-cb28-4629-b7f6-8980484c42f9)

**Physical Activity Levels**

-   **Activity Rates**: A robust **77.5%** of individuals engage in physical activity, which is positive for public health outcomes.
-   
    ![13](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/a40d434a-63eb-40ce-95b6-82875cf6a9d5)
    
**General Health Status**

-   **Self-reported Health**: The most common health ratings are "Very good" (35.8%), followed by "Good" (29.3%) and "Excellent" (21.0%), suggesting that many individuals perceive their health positively.

    ![14](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/f1c04ba4-7d5d-4864-a633-50cdcb75d106)

**Sleep Patterns**

-   **Sleep Duration**: The average sleep duration is **7.1** hours per day, which is within the recommended range for adults.

    ![15](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/ab23a3f3-4038-41e6-8b11-44d2b9402d1f)

**Asthma and Kidney Disease**

-   **Asthma Prevalence**: **13.4%** of the surveyed group report having asthma.

    ![16](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/5dd8c12a-6140-413b-b835-dc457808c601)

-   **Kidney Disease**: **3.7%** are affected by kidney disease, highlighting specific health conditions within the population.

    ![17](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/7481a142-83c5-452d-a8d0-1bdb5f972a8b)

**Skin Cancer Concerns**

-   **Skin Cancer Prevalence**: **9.3%** of the participants have been diagnosed with skin cancer, which is significant for oncological and dermatological health considerations.

    ![18](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/45cfa4bb-261b-445d-a40f-53772e3e6e42)

## **Expanded Insights and Implications:**

This comprehensive analysis provides an in-depth look at various health indicators across a diverse demographic. The findings reveal significant correlations between lifestyle choices such as smoking and alcohol consumption with chronic diseases like heart disease. The data on physical and mental health underscores the need for integrated health services that address both physical and psychological aspects of well-being.

Additionally, the variations in health status across different age groups and races suggest the necessity for tailored health interventions that consider the unique needs of these subgroups. The overall high engagement in physical activity is encouraging, yet the presence of chronic conditions like diabetes and heart disease highlights the complexity of health dynamics in populations.

Further detailed studies are necessary to explore the causal relationships and to develop targeted strategies for improving health outcomes based on these insights. This EDA not only sheds light on the current state of health in this cohort but also underscores the critical areas where healthcare resources and preventive measures can be effectively directed.

## **Predictive Modeling and Analysis:**

Using Logistic Regression, RandomForestRegressor, and SVM, the project constructs predictive models to estimate the risk of heart disease based on the identified factors. Each model is evaluated to determine its effectiveness in capturing the complex relationships within the data and predicting heart disease accurately. The models are compared based on their performance metrics, including accuracy and ROC curves, to select the most suitable model for practical applications.

**The accuracy achieved by the models is as follows:**

\- **Logistic Regression: 0.913**

\- **RandomForestRegressor: 0.903**

\- **SVM: 0.911**

These accuracy metrics highlight the capabilities of each modeling approach in the context of this specific dataset. Logistic Regression shows the highest accuracy, suggesting its robustness and efficiency in handling binary classification problems like predicting heart disease. The slight variation in the performance among the models provides valuable insights into how different algorithms manage the nuances of the dataset, with Logistic Regression slightly outperforming the others in overall accuracy. This information is crucial for guiding the choice of model for deployment in real-world applications where accuracy is paramount.

![20](https://github.com/omarfayez-151/Bitcoin-Price-Forecasting/assets/134233189/27b8df81-8dc0-4c0a-a065-15e322a75770)

## **Conclusion and Implications:**

This project not only highlights significant factors associated with heart disease but also demonstrates the power of machine learning in enhancing our understanding and ability to predict health outcomes. The insights gained from the detailed analysis are crucial for shaping future healthcare strategies and interventions aimed at reducing the prevalence and impact of heart disease. The predictive models developed herein offer a foundation for further research and development in healthcare analytics, showcasing a path forward in the fight against one of the leading causes of death globally.

By addressing these key questions and leveraging advanced analytics techniques, this notebook provides actionable insights that could inform healthcare providers, policymakers, and individuals about effective strategies to mitigate heart disease risk.
