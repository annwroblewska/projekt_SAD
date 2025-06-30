'''  próba na projekt SAD '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from statsmodels.imputation.mice import MICEData    # pip install statsmodels
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

DANE_RAW = 'silownia_new.csv'

def main():
    
    df_raw = pd.read_csv(DANE_RAW)

    # sprawdzenie danych
    print(df_raw.info())
    #assert 0, 'Przerwano działanie'
    #print(df_raw.describe())
    #print(df_raw.head())

    df = df_raw.copy()

    sprawdzenie_poprawności_danych(df)

    sprawdzanie_brakujących_wartości(df)
    # BMI można uzupełnić na podstawie wzrostu i wagi
    # Age można uzupełnić na podstawie Resting_BPM - max tętno zależy od wieku
    # Workout_type można uzupełnić na podstawie pozostałych danych
    
    df_compl = uzupełnianie_brakujących_wartości(df)
    # Age słabe wyniki uzupełniania danych ze wzoru
    # Age słabe wyniki uzupełniania danych algorytmami ML
    # Trafność przewidywania na danych testowych R2 na poziomie 0.1
    # Workout_type słabe wyniki uzupełniania danych algorytmami ML
    # Trafność przewidywania na danych testowych R2 na poziomie 0.3

    sprawdzanie_brakujących_wartości(df)

    sprawdzenie_parametrów_statystycznych(df)

    df_outliers = sprawdzanie_odstających_wartości(df)
    df_clear = df[df_outliers['is_outlaier'] == False].copy()  # odfiltrowanie wartości odstających
    print('Rozmiar danych przed odfiltrowaniem: ', df.shape)
    print('Rozmiar danych po odfiltrowanu: ', df_clear.shape)

    sprawdzenie_relacji_w_danych(df_clear)

    sprawdzanie_korelacji_w_danycj(df_clear)
    

    return []


def sprawdzenie_poprawności_danych(df):

    # po zapoznaniu się ze specyfiką danych:
    # wszystkie wartości liczbowe nie powinny mieć wartości ujemnych
    # niektóre wartości można sprawdzać na podstawie wiedzy medycznej
    # np. ekstremalnie niskie tętno nie powinno być możliwe w trakcie wysiłku fizycznego
    # mając dane tętna powinien być spełniony warunek:
    #    Resting_BPM <= Avg_BPM <= Max_BPM
    # Workout_Frequency (days/week) nie może być większe niż 7
    # Experience_Level kategoria z założenia musi być z zakresu od 1 do 3 typu int
    # Age powinien być poniżej 120 
    # wartość BMI wynika z wagi i wzrostu według wzoru

    print('-----------------------------------------------')
    print('Sprawdzenie poprawności danych')
    print('-----------------------------------------------')
    
    print('    Sprawdzanie wartości w kolumnach liczbowych')
    temp_all = False
    for column in df.columns:
        #pandas.api.types.is_numeric_dtype(arr_or_dtype)        
        if pd.api.types.is_numeric_dtype(df[column]):
            temp = (df[column] < 0).any()
            if temp == True:
                temp_all = True
        print('        Kolumna',column,', czy występują watrrości ujemne? :', temp)
    print('      Czy wykryto wartości ujemne w kolumnach liczbowych:', temp_all)

    temp = (df['Resting_BPM'] > df['Avg_BPM']).any() and (df['Avg_BPM'] > df['Max_BPM']).any()
    print('  Czy spełniony jest warunek: Resting_BPM <= Avg_BPM <= Max_BPM ? :', ~temp)
    print('  Czy Workout_Frequency <= 7 ? :', ~(df['Workout_Frequency (days/week)'] > 7).any() )
    print('  Czy Experience_Level jest typu integer ? :', pd.api.types.is_integer_dtype(df['Experience_Level'].dtype))
    temp = (df['Experience_Level'] < 1).any() or (df['Experience_Level'] > 3).any()
    print('  Czy Experience_Level jest z zakresu od 1 do 3 ? :', ~temp)
    print('  Czy wartości w Age są poniżej 120 ? :', ~(df['Age'] >= 120).any()  )
    temp = ( df['BMI'] == ( df['Weight (kg)'] / (df['Height (m)'] ** 2) ) ).any()
    print('  Czy wartości w BMI są poprawnie przeliczone ? :', temp)

    print('-----------------------------------------------')
    print('Zakończono sprawdzenie poprawności danych')
    print('-----------------------------------------------')

    return []

def sprawdzanie_brakujących_wartości(df):

    print('------------------------------------------')
    print('Sprawdzanie brakujących wartości.')
    print('------------------------------------------')

    #msno.matrix(df)
    #msno.heatmap(df)
    #msno.dendrogram(df)

    # sprawdzenie ilości NaN
    print('Ilość Nan w danych:')
    print(df.isna().sum().sort_values(ascending=False))
    print('------------------------------------------')
    print('Ilość wierszy z NaN:')
    print(df.isna().any(axis=1).sum())
    print('------------------------------------------')
    print('Ilość kolumn w danych:', df.columns.size)
    print('Ilość Nan w wierszach:')
    print(df.isna().sum(axis=1).value_counts())

    return []

def uzupełnianie_brakujących_wartości(df):

    df_z_brakami = df.copy()

    df_temp = pd.DataFrame(df[['BMI', 'Weight (kg)', 'Height (m)']])
    df_temp['BMI_calc'] = df_temp['Weight (kg)'] / (df_temp['Height (m)'] ** 2)

    sns.scatterplot(data=df_temp, x='BMI', y='BMI_calc')
    plt.show()

    print('------------------------------------------')
    print('Uzupełnianie brakujących wartości.')
    print('------------------------------------------')
    # uzupełnianie BMI
    filtr_4_BMI = df['BMI'].isna()
    df.loc[filtr_4_BMI, 'BMI'] = df.loc[filtr_4_BMI, 'Weight (kg)'] / (df.loc[filtr_4_BMI, 'Height (m)'] ** 2) 
    
    # porównanie histogramów dla BMI
    sns.histplot(data=df_z_brakami, x='BMI')
    sns.histplot(data=df, x='BMI')
    plt.show()

    # uzupełnianie wieku
    # Wzór Tanaki:  max_tętno =  208 - (0.7 x wiek)
    # wiek = (208 - max_tętno) / 0.7
    filtr_4_Age = df['Age'].isna()

    #plt.scatter(df.loc[filtr_4_Age, 'Resting_BPM' ].values, df.loc[filtr_4_Age, 'Age' ].values)
    sns.scatterplot(data=df, x='Avg_BPM', y='Age', hue='Workout_Type')
    plt.show()

    df.loc[filtr_4_Age, 'Age' ] =  (208 - df.loc[filtr_4_Age, 'Max_BPM' ].values ) / 0.7 

    # porównanie histogramów dla BMI
    sns.histplot(data=df_z_brakami, x='Age')
    sns.histplot(data=df, x='Age')
    sns.histplot(data=df[filtr_4_Age], x='Age')
    plt.show()

    # uzupełnianie wieku przez ML
    df_dummy_1 = pd.get_dummies(df['Gender'])
    df_dummy_2 = pd.get_dummies(df['Workout_Type'])
    df_temp = df.drop(labels=['Gender', 'Workout_Type'],axis=1)
    df_4_impute = pd.concat( [df_temp, df_dummy_1, df_dummy_2], axis=1).dropna().copy()
    #df_4_impute = df.dropna().copy()

    model_KNN = KNeighborsRegressor(n_neighbors=3)
    model_RFc = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    y = df_4_impute[['Age']].values.reshape((-1,))
    X = df_4_impute.drop(labels='Age',axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model_KNN.fit(X_train, y_train)
    model_RFc.fit(X_train, y_train)
    print('score, model KNN dla danych uczących:', model_KNN.score(X_train, y_train) )
    print('score, model KNN dla danych testowych:', model_KNN.score(X_test, y_test) )
    print('score, model RFc dla danych uczących:', model_RFc.score(X_train, y_train) )
    print('score, model RFc dla danych testowych:', model_RFc.score(X_test, y_test) )


    # uzupełnianie Workout_Type
    df_dummy = pd.get_dummies(df['Gender'])
    df_temp = df.drop(labels='Gender',axis=1)
    df_4_impute = pd.concat( [df_temp, df_dummy], axis=1).dropna().copy()
    #df_4_impute = df.dropna().copy()

    print(df_4_impute.info())

    model_KNN = KNeighborsClassifier(n_neighbors=3)
    model_RFc = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    y = df_4_impute[['Workout_Type']].values.reshape((-1,))
    X = df_4_impute.drop(labels='Workout_Type',axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    model_KNN.fit(X_train, y_train)
    model_RFc.fit(X_train, y_train)
    print('score, model KNN dla danych uczących:', model_KNN.score(X_train, y_train) )
    print('score, model KNN dla danych testowych:', model_KNN.score(X_test, y_test) )
    print('score, model RFc dla danych uczących:', model_RFc.score(X_train, y_train) )
    print('score, model RFc dla danych testowych:', model_RFc.score(X_test, y_test) )

    #assert 0, "Przerwano wykonywanie"

    # mice_data = MICEData(df_4_impute)    # Tworzymy obiekt MICEData
    # for i in range(5):  # Iteracyjna imputacja brakujących wartości
    #     mice_data.update_all()

    # df_imputed = mice_data.data     # Uzyskanie imputowanego DataFrame
    # df['Gender'] = df_imputed['Gender'].copy()

    return df.copy()


def sprawdzenie_parametrów_statystycznych(df):

    print('-------------------------------------------------------')
    print('Sprawdzenie parametrów_statystycznych')
    print('-------------------------------------------------------')
    print(df.describe())
    print(df[['Workout_Type', 'Gender']].describe())
    print('  Kategorie w kolumnie Gender:', df['Gender'].unique())
    print('  Kategorie w kolumnie Workout_Type:', df['Workout_Type'].unique())
    print('  Kategorie w kolumnie Experience_Level:', df['Experience_Level'].unique())


    return []

def sprawdzanie_odstających_wartości(df):

    print('-------------------------------------------------------')
    print('Sprawdzenie odstających wartości w danych')
    print('-------------------------------------------------------')

    # określenie kolumn numerycznych i kategorycznych
    column = []
    column_numeric = []
    column_categorical = []
    temp_all = False
    for col in df.columns:
        column.append(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            column_numeric.append(col)
        else:
            column_categorical.append(col)
    column_categorical.append('Experience_Level')
    column_numeric.remove('Experience_Level')
    print('    Kolumny z danymi numerycznymi:', column_numeric)
    print('    Kolumny z danymi kategorycznymi:', column_categorical)

    df_outlier = pd.DataFrame()
    for col in column_numeric:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        icq = q3-q1
        col_new_name = '_outliers_' + col
        df_outlier[ col_new_name ] = (df[col]<(q1-1.5*icq)) | (df[col]>(q3+1.5*icq)) 
    
    df_outlier[ 'is_outlaier' ] = df_outlier.any(axis=1)

    print('-------------------------------------------------------')
    print('  Ilości wykrytych outliers:')
    print(df_outlier.sum(axis=0))
    print('-------------------------------------------------------')
    print('  Ilość wykrytych odstających wartości:', df_outlier['is_outlaier'].sum())

    #assert 0, 'Przerwana działanie'


    return df_outlier

def sprawdzenie_relacji_w_danych(df):

    df_clean = df.dropna().copy()

    print('-------------------------------------------------------')
    print('Sprawdzenie relacji w danych')
    print('-------------------------------------------------------')

    column = []
    column_numeric = []
    column_categorical = []
    temp_all = False
    for col in df.columns:
        column.append(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            column_numeric.append(col)
        else:
            column_categorical.append(col)
    column_categorical.append('Experience_Level')
    column_numeric.remove('Experience_Level')
    print('    Kolumny z danymi numerycznymi:', column_numeric)
    print('    Kolumny z danymi kategorycznymi:', column_categorical)

    sns.violinplot(data=df_clean, x='Workout_Type', y='Resting_BPM', hue='Gender')
    plt.show()
    sns.violinplot(data=df_clean, x='Workout_Type', y='Avg_BPM', hue='Gender')
    plt.show()
    sns.violinplot(data=df_clean, x='Workout_Type', y='Max_BPM', hue='Gender')
    plt.show()
    sns.countplot(data=df_clean, x='Workout_Type', hue='Experience_Level')
    plt.show()

    #sns.histplot(data=df[column_numeric])
    #plt.show()
    #sns.heatmap(data=df[column_numeric])
    #plt.show()


    return []

def sprawdzanie_korelacji_w_danycj(df):

    # co wpływa na tętno spoczynkowe ?

    column = []
    column_numeric = []
    column_categorical = []
    temp_all = False
    for col in df.columns:
        column.append(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            column_numeric.append(col)
        else:
            column_categorical.append(col)
    column_categorical.append('Experience_Level')
    column_numeric.remove('Experience_Level')
    print('    Kolumny z danymi numerycznymi:', column_numeric)
    print('    Kolumny z danymi kategorycznymi:', column_categorical)

    # podejście jednowymiarowe
    corr = pd.DataFrame()
    corr['corr Resting_BPM'] = df[column_numeric].corr()[['Resting_BPM']]
    corr['abs |corr Resting_BPM|'] = corr['corr Resting_BPM'].abs()
    print(corr)

    # podejście wielowymiarowe - algorytm ML wykorzystany do określenia zależności
    df_4_ml = df.copy()   #.dropna()
    df_dummy_1 = pd.get_dummies(df_4_ml['Gender'])
    df_dummy_2 = pd.get_dummies(df_4_ml['Workout_Type'])
    df_temp = df_4_ml.drop(labels=['Gender', 'Workout_Type'],axis=1)
    df_4_ml = pd.concat( [df_temp, df_dummy_1, df_dummy_2], axis=1).copy()

    corr_ = pd.DataFrame()
    corr_['corr Resting_BPM'] = df_4_ml.corr()[['Resting_BPM']]
    corr_['abs |corr Resting_BPM|'] = corr_['corr Resting_BPM'].abs()

    y = df_4_ml['Resting_BPM'].values.reshape((-1,))
    X = df_4_ml.drop(labels=['Resting_BPM'], axis=1).values

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X,y)
    print('Model score:', model.score(X,y))
    importances = model.feature_importances_
    feature_names = df_4_ml.drop(labels=['Resting_BPM'], axis=1).columns.tolist()
    forest_importances = pd.Series(importances, index=feature_names, name='ML_imp')
    print(forest_importances)
    forest_importances_df = pd.DataFrame(forest_importances) #, columns='ML_imp')
    #plt.show()
    print(forest_importances_df)
    #forest_importances.plot.bar()
    #corr.plot.bar()
    
    #sns.barplot(data=forest_importances_df, x='ML_imp')
    #sns.barplot(data=corr_, x='abs |corr Resting_BPM|')
    #plt.show()

    corr_['ML_importance'] = corr_['corr Resting_BPM']
    for index,row in corr_.iterrows():
        if index == 'Resting_BPM':
            corr_.loc[index, 'ML_importance'] = 1
        else:
            corr_.loc[index, 'ML_importance'] = forest_importances[index]

    print(corr_)

    corr_[['abs |corr Resting_BPM|', 'ML_importance']].plot.bar(title='Zależność Resting_BPM od innych zmiennych (korelacja)')
    plt.show()
    

    return []

if __name__ == '__main__':
    main()
