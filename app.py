import streamlit as st
from streamlit_option_menu import option_menu

import pickle
from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

plt.rcParams["figure.autolayout"] = False

img = Image.open('download.jpeg')
st.set_page_config(page_title = 'FutureX', page_icon = img)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">FutureX</h1>', unsafe_allow_html=True)


st.markdown('<h2 style="text-align: centre;">"Cracking the Campus Code: Exploring Placement Patterns and Trends"</h2>', unsafe_allow_html=True)


# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Page of Project",
        options = ["Project Introduction", "Data Input", "Round-wise Placement", "Model Analysis"],
)

if selected == "Project Introduction":
    st.subheader(f"The page provides us a breif summary of the overall code we have employed in our project")
#if selected == "Real Time working":
    #st.title(f"Welcome!")
if selected == "Working":
    st.title("Code Demo")

if selected == "Project Introduction":
    st.subheader('...')

    #user_input = st.text_input('Enter Placement Score')

    df = pd.read_excel("Student-Employability-Datasets.xlsx")

    df.head() #st.write(df.head())

    df.describe() #st.write(df.describe())

    new_df = df.drop(
        columns = [
            'CLASS',
            'Name of Student',
            'Student Performance Rating'
        ]
    )

    total = pd.DataFrame({
            'Skills' : new_df.columns,
            'Total Value' : new_df.sum()
        })

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(7.5)
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style="darkgrid", palette="deep")

    ax = sns.barplot(
            y = 'Skills',
            x='Total Value',
            data = total.sort_values(
                'Total Value',
                ascending = False
            )
        )
    ax.set_title('Summation of Values per skill')
    st.title('Summation of Values per Skill')
    st.pyplot(fig)
    fig.patch.set_facecolor('ghostwhite')

    df_employed = df.loc[df['CLASS'] == 'Employable']

    df_employed = df_employed.drop(columns = ['Name of Student','CLASS'])

    ave_skills = pd.DataFrame({
            'Average Score' : df_employed.mean()
        })

    ave_skills #st.write(ave_skills)

    df_employed.head()

    pie_data = ave_skills.drop(index = ['Student Performance Rating'], axis = 0)
    pie_data

    palette_color = sns.color_palette(palette = 'deep')
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_facecolor('ghostwhite')

    plt.pie(
        pie_data['Average Score'],
        labels = pie_data.index,
        colors = palette_color,
        autopct = '%.0f%%'
    )
    centre_circle = plt.Circle((0, 0), 0.72, fc='white')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Percentage Distribution of Skills for Employable on Average')
    st.title('Percentage Distribution of Skills for Employable on Average')
    st.pyplot(fig)

    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style="darkgrid", palette="deep")

    sns.barplot(
        y = ave_skills.index,
        x= 'Average Score',
        data = ave_skills.sort_values(
            'Average Score',
            ascending = False
        )
    )

    plt.xticks(np.arange(0,5.25,0.25))
    plt.title('Average Score of the Employable Students per Category')
    st.title('Average Score of the Employable Students per Category')
    st.pyplot(fig)

    df_ave = pd.DataFrame({
        'Student' : df['Name of Student'].loc[df['CLASS'] == 'Employable'],
        'Average Score': df.loc[df['CLASS'] == 'Employable']\
        ._get_numeric_data().mean(axis = 1)
    })
    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style='darkgrid', palette = 'dark')

    ax = sns.histplot(
        x = 'Average Score',
        data = df_ave
    )
    df_ave.plot(
        kind='kde',
        ax=ax,
        secondary_y=True,
        color = 'orange'
    )

    plt.xticks(np.arange(2,6.25,0.25))
    st.title('Mean Scores per Student Histogram for Employable')
    plt.title('Mean Scores per Student Histogram for Employable')
    st.pyplot(fig)


    df_less = df.loc[df['CLASS'] == 'LessEmployable']\
    .drop(columns = ['CLASS', 'Name of Student'])
    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    less_ave_skills = pd.DataFrame({
        'Average' : df_less.mean()
    })

    sns.set_theme(style="darkgrid", palette="deep")

    sns.barplot(
        y = less_ave_skills.index,
        x= 'Average',
        data = less_ave_skills.sort_values(
            'Average',
            ascending = False
        )
    )

    plt.xticks(np.arange(0,5.25,0.25))
    st.title('Average Score of the LESS Employable Students per Category')
    plt.title('Average Score of the LESS Employable Students per Category')
    st.pyplot(fig)

    df_ave_less = pd.DataFrame({
        'Student' : df['Name of Student'].loc[
            df['CLASS'] == 'LessEmployable'
        ],
        'Average Score': df.loc[
            df['CLASS'] == 'LessEmployable'
        ]._get_numeric_data().mean(axis = 1)
    })

    df_ave_less.head()

    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style='darkgrid', palette = 'dark')

    ax = sns.histplot(
        x = 'Average Score',
        data = df_ave_less
    )

    df_ave_less.plot(
        kind='kde',
        ax=ax,
        secondary_y=True,
        color = 'darkorange'
    )

    plt.xticks(np.arange(2,6.25,0.25))
    st.title('Mean Scores per Student Histogram for LESS Employable')
    plt.title('Mean Scores per Student Histogram for LESS Employable')
    st.pyplot(fig)

    st.title('LESS EMPLOYABLE STUDENTS')
    st.write(df_less.loc[df_ave_less['Average Score'] > 3.5].head(10))

    st.title('EMPLOYABLE STUDENTS')
    st.write(df_employed.loc[df_ave['Average Score'] > 3.5].head(10))

    sns.pairplot(
        vars = [
            'GENERAL APPEARANCE',
            'MANNER OF SPEAKING',
            'PHYSICAL CONDITION',
            'MENTAL ALERTNESS',
            'SELF-CONFIDENCE',
            'ABILITY TO PRESENT IDEAS',
            'COMMUNICATION SKILLS'
        ],
        hue = 'CLASS',
        kind = 'scatter',
        data = df
    )
    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    ax = sns.heatmap(
        df.drop(
            columns = [
                'Name of Student',
                'CLASS',
                'Student Performance Rating'
            ]
        ).corr().round(2),
        annot=True,
        vmin = -1,
        vmax = 1,
        center = 0,
        cmap ='YlGnBu'
    )

    plt.xticks(rotation = 45)
    st.title('Correlation Matrix for the Dataset')
    plt.title("Correlation Matrix for the Dataset")
    st.pyplot(fig)

    st.subheader('These tables and graphs demonstrates to us how the test values and train values have been utilized to obtain graphs for the placement trends')

if selected == "Data Input":
    st.header("please Input the necessary placement scores")
    user_data_list = []
    entry_counter = 1
    while True:
        name_key = f"Name_{entry_counter}"
        General_Appearance_key = f"General Appearance_{entry_counter}"
        Manner_of_Speaking_key = f"Manner of Speaking_{entry_counter}"
        Physical_Condition_key = f"Physical Condition_{entry_counter}"
        Mental_Alertness_key = f"Mental Alertness_{entry_counter}"
        Self_Confidence_key = f"Self Confidence_{entry_counter}"
        Ability_to_Present_Ideas_key = f"Ability to Present Ideas_{entry_counter}"
        Communication_Skills_key = f"Communication Skills_{entry_counter}"
        submit_key = f"Submit_{entry_counter}"
        add_more_data_key = f"Add More Data_{entry_counter}"

        user_name = st.text_input("Name", key=name_key)
        General_Appearance = st.number_input("General Appearance", key = General_Appearance_key)
        Manner_of_Speaking = st.number_input("Manner of Speaking", key = Manner_of_Speaking_key)
        Physical_Condition = st.number_input("Physical Condition", key = Physical_Condition_key)
        Mental_Alertness = st.number_input("Mental Alertness", key = Mental_Alertness_key)
        Self_Confidence = st.number_input("Self Confidence", key = Self_Confidence_key)
        Ability_to_Present_Ideas = st.number_input("Ability to Present Ideas", key = Ability_to_Present_Ideas_key)
        Communication_Skills = st.number_input("Communication Skills", key = Communication_Skills_key)

        if st.button("Submit", key=submit_key):
            new_entry = {"User Name": user_name, "General Appearance": General_Appearance, "Manner of Speaking": Manner_of_Speaking, "Physical Condition": Physical_Condition, "Mental Alertness": Mental_Alertness, "Self Confidence": Self_Confidence, "Ability to Present Ideas": Ability_to_Present_Ideas, "Communication Skills": Communication_Skills}
            user_data_list.append(new_entry)
            #user_data.to_excel("User_data.xlsx", index=False)
            st.success("Data submitted successfully!")
        add_more_data = st.button("Add More Data", key=add_more_data_key)
        if not add_more_data:
            break
        entry_counter += 1
    user_data_df = pd.DataFrame(user_data_list)
    user_data_df.to_excel("User_data.xlsx", index=False)

if selected == "Round-wise Placement":

    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score

    df = pd.read_excel("Student-Employability-Datasets.xlsx")

    df['CLASS'] = df['CLASS'].str.replace('LessEmployable', '0')
    df['CLASS'] = df['CLASS'].str.replace('Employable', '1')

    df['CLASS'].dtypes
    df['CLASS']

    df['CLASS'] = df['CLASS'].astype(float)
    df

    # Separate features and target
    X = df.drop(columns=["CLASS"])  # Features
    y = df["CLASS"]  # Target

    chosen_feature = ['Student Performance Rating','GENERAL APPEARANCE','ABILITY TO PRESENT IDEAS','MENTAL ALERTNESS','MANNER OF SPEAKING','PHYSICAL CONDITION','SELF-CONFIDENCE','COMMUNICATION SKILLS']  # Replace with the name of the feature you want to choose
    X_selected = X[chosen_feature]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    clf = RandomForestClassifier()
    clf.fit(X_train.values,y_train)
    y_pred = clf.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(random_state=0)

    #without scaling
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


    model1 = RandomForestClassifier(random_state=42)
    model2 = KNeighborsClassifier(n_neighbors=1)
    model3 = DecisionTreeClassifier(random_state=0)

    ensemble = VotingClassifier(
        estimators = [('model1', model1), ('model3', model3), ('model2', model2)],
        voting='hard'
    )

    ensemble = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader(f'We have achieved an accuracy of: {accuracy}')


    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    # Create a RandomForestClassifier
    rf_classifier = RandomForestClassifier()

    # Define the search space for hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    test_accuracy = best_model.score(X_test, y_test)
    print("Best Hyperparameters:", best_params)
    print("Test Accuracy:", test_accuracy)

    # Assuming 'X' is your original dataset with 8 features
    # Selecting only the first 4 features
    #X_subset = df.iloc[:, :4]

    #Passing values to round 01 of the placement
    chosen_feature = ['GENERAL APPEARANCE','MENTAL ALERTNESS','PHYSICAL CONDITION','SELF-CONFIDENCE']  # Replace with the name of the feature you want to choose
    X_selected = X[chosen_feature]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import train_test_split

    model1 = RandomForestClassifier(random_state=42)
    model2 = KNeighborsClassifier(n_neighbors=1)
    model3 = DecisionTreeClassifier(random_state=0)


    ensemble = VotingClassifier(
        estimators=[('model1', model1), ('model3', model3)],
        voting='hard'  # Use 'soft' for weighted voting based on probabilities
    )

    ensemble = RandomForestClassifier(
        n_estimators=100, # Number of base models
        random_state=42
    )

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {accuracy}")

    y_train_pred = ensemble.predict(X_train)

    for i in range(len(y_train)):
        st.write(f"Actual: {y_train.iloc[i]}, Predicted: {y_train_pred[i]}")

    # Assuming you have y_test and y_pred from previous code
    y_pred_zeros = y_test[y_pred == 0]

    # Count the number of instances with predicted value 0
    num_zeros = len(y_pred_zeros)

    st.write(f"Number of instances with predicted value 0: {num_zeros}")

    # Create a DataFrame with X_test, y_test, and y_pred
    test_data = pd.DataFrame(data=X_test, columns=X_test.columns)
    test_data["actual"] = y_test
    test_data["predicted"] = y_pred

    # Group instances with predicted value 0
    grouped_less_emp_round1 = test_data[test_data["predicted"] == 0]
    r1_less_emp = pd.DataFrame(grouped_less_emp_round1)
    r1_less_emp.to_excel("Round_1_less_emp.xlsx", index=False)
   

    # Group instances with predicted value 1
    grouped_emp_round1 = test_data[test_data["predicted"] == 1]
    r1_emp = pd.DataFrame(grouped_emp_round1)
    r1_emp.to_excel("Round_1_emp.xlsx", index=False)

    # Display the grouped instances
    st.write(grouped_less_emp_round1)

    ave_skills = pd.DataFrame({
    'Average Score' : grouped_less_emp_round1.mean()
    })

    ave_skills

    ave_skills = ave_skills.drop(index = ['actual'], axis = 0)
    ave_skills = ave_skills.drop(index = ['predicted'], axis = 0)

    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style="darkgrid", palette="deep")

    sns.barplot(
        y = ave_skills.index,
        x= 'Average Score',
        data = ave_skills.sort_values(
            'Average Score',
            ascending = False
        )
    )

    plt.xticks(np.arange(0,5.25,0.25))
    plt.title('Average Score of the Less Employable Students per Category in ROUND 1')
    st.pyplot(fig)

    #Round2 
    #df = pd.read_excel("Round_1_emp.xlsx")
    grouped_emp_round1["Student Performance Rating"] = df["Student Performance Rating"]
    grouped_emp_round1["COMMUNICATION SKILLS"] = df["COMMUNICATION SKILLS"]
    grouped_emp_round1["MANNER OF SPEAKING"] = df["MANNER OF SPEAKING"]
    grouped_emp_round1["ABILITY TO PRESENT IDEAS"] = df["ABILITY TO PRESENT IDEAS"]
    grouped_emp_round1

    chosen_feature = ['MANNER OF SPEAKING', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS', 'Student Performance Rating']  # Replace with the name of the feature you want to choose
    X_selected = X[chosen_feature]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import train_test_split

    model1 = RandomForestClassifier(random_state=42)
    model2 = KNeighborsClassifier(n_neighbors=1)
    model3 = DecisionTreeClassifier(random_state=0)


    ensemble = VotingClassifier(
        estimators=[('model1', model1), ('model3', model3), ('model2', model2)],
        voting='hard'  # Use 'soft' for weighted voting based on probabilities
    )

    ensemble = RandomForestClassifier(
        n_estimators=100, # Number of base models
        random_state=42
    )

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {accuracy}")
    y_train_pred = ensemble.predict(X_train)

    for i in range(len(y_train)):
        print(f"Actual: {y_train.iloc[i]}, Predicted: {y_train_pred[i]}")

    # Assuming you have y_test and y_pred from previous code
    y_pred_zeros = y_test[y_pred == 0]

    # Count the number of instances with predicted value 0
    num_zeros = len(y_pred_zeros)

    st.write(f"Number of instances with predicted value 0: {num_zeros}")

    # Create a DataFrame with X_test, y_test, and y_pred
    test_data = pd.DataFrame(data=X_test, columns=X_test.columns)
    test_data["actual"] = y_test
    test_data["predicted"] = y_pred

    # Group instances with predicted value 0
    grouped_less_emp_round2 = test_data[test_data["predicted"] == 0]


    # Group instances with predicted value 1
    grouped_emp_round2 = test_data[test_data["predicted"] == 1]
    r2_emp = pd.DataFrame(grouped_emp_round2)
    r2_emp.to_excel("Round_2_emp.xlsx", index=False)


    # Display the grouped instances
    print(grouped_less_emp_round2)

    ave_skills_lemp_r2 = pd.DataFrame({ 'Average Score' : grouped_less_emp_round2.mean() })
    ave_skills_lemp_r2

    ave_skills_lemp_r2 = ave_skills_lemp_r2.drop(index = ['actual'], axis = 0)
    ave_skills_lemp_r2 = ave_skills_lemp_r2.drop(index = ['predicted'], axis = 0)

    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style="darkgrid", palette="deep")

    sns.barplot(
        y = ave_skills_lemp_r2.index,
        x= 'Average Score',
        data = ave_skills_lemp_r2.sort_values(
            'Average Score',
            ascending = False
        )
    )

    plt.xticks(np.arange(0,5.25,0.25))
    plt.title('Average Score of the Less Employable Students per Category in ROUND 2')
    st.pyplot(fig)

    #Fianl Employed Students Score
    # Group instances with predicted value 1
    grouped_emp_round2 = test_data[test_data["predicted"] == 1]

    # Display the grouped instances
    print(grouped_emp_round2)

    grouped_emp_round2["GENERAL APPEARANCE"] = df["GENERAL APPEARANCE"]
    grouped_emp_round2["MENTAL ALERTNESS"] = df["MENTAL ALERTNESS"]
    grouped_emp_round2["PHYSICAL CONDITION"] = df["PHYSICAL CONDITION"]
    grouped_emp_round2["SELF-CONFIDENCE"] = df["SELF-CONFIDENCE"]
    grouped_emp_round2

    ave_skills_emp = pd.DataFrame({ 'Average Score' : grouped_emp_round2.mean() })
    ave_skills_emp


    ave_skills_emp = ave_skills_emp.drop(index = ['actual'], axis = 0)
    ave_skills_emp = ave_skills_emp.drop(index = ['predicted'], axis = 0)

    fig = plt.figure(figsize=(15,7.5))
    fig.patch.set_facecolor('ghostwhite')

    sns.set_theme(style="darkgrid", palette="deep")

    sns.barplot(
        y = ave_skills_emp.index,
        x= 'Average Score',
        data = ave_skills_emp.sort_values(
            'Average Score',
            ascending = False
        )
    )

    plt.xticks(np.arange(0,5.25,0.25))
    plt.title('Average Score of the Employable Students')
    st.pyplot(fig)

if selected == "Model Analysis":
    st.header("Graphical Representation of the Model and its Accurace")
    classifiers = [
    'RandomForestClassifier',
    'KNeighborsClassifier',
    'LogisticRegressionCV',
    'CategoricalNB',
    'SGDClassifier',
    'BernoulliNB',
    'DecisionTreeClassifier',
    'NuSVC',
    'MultinomialNB',
    'BernoulliNB',
    'CategoricalNB',
    'Perceptron',
    'DecisionTreeRegressor',
    'VotingClassifier',
    'BaggingClassifier',
    'VotingClassifier(GradientBoostingClassifier, AdaBoostClassifier)'
    ]

    scores = [
        0.909547739,
        0.894472362,
        0.594639866,
        0.587939698,
        0.587939698,
        0.556113903,
        0.909547739,
        0.850921273,
        0.556113903,
        0.556113903,
        0.587939698,
        0.443886097,
        -1,  # You have a missing score for DecisionTreeRegressor
        0.911222781,
        0.909547739,
        0.817420436
    ]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(classifiers, scores, color='skyblue')
    ax.set_xlabel('Accuracy Score')
    ax.set_title('Classifier Performance')
    ax.set_xlim(0, 1)  # Set the x-axis limits between 0 and 1 for accuracy scores

    # Display the scores on the bars
    for i, score in enumerate(scores):
        if score != -1:
            ax.text(score, i, f' {score:.4f}', va='center', fontsize=12, color='black')

    ax.invert_yaxis()
      # Invert the y-axis to display the highest score at the top

    ax.set_facecolor('none')

    st.pyplot(fig)
