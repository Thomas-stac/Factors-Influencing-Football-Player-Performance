import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample

ST = pd.read_csv('Base_de_données.csv')

#Affiche le nombre de lignes et colonnes
print("Base de données des joueurs offensifs de 2023:", ST.shape)

#J'ai fait varier mes suppressions de variables afin de voir si l'impact de certaines n'étaient pas contre intuitive comme pour 'club_jersey_number',
#néanmoins elles n'ont jamais eu d'impact dans mes modèles.


#Suite à une première analyse de nombreuses variables sont supprimées. Les raisons sont expliquées dans le travail écrit.
colonnes_a_supprimer = ['player_url' ,'fifa_update_date','long_name','short_name', 'league_name', 'club_name', 'club_jersey_number', 'fifa_version',
                        'nationality_name', 'nation_jersey_number', 'player_face_url', 'club_joined_date','club_contract_valid_until_year',
                        'dob', 'club_loaned_from', 'nation_team_id', 'nation_position', 'player_tags', 'player_traits', 'goalkeeping_speed', 
                        'release_clause_eur', 'player_positions', 'real_face', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                        'goalkeeping_positioning', 'goalkeeping_reflexes', 'lwb','ldm','cdm','rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                        'lm', 'lcm', 'cm', 'rcm','rm','lw', 'rf','lf','rs','ls','rw', 'lam', 'cam', 'st','cf', 'ram', 'club_position', 'body_type',
                        'player_id'
                        ]

data_ST = ST.drop(columns=colonnes_a_supprimer)


""" C'est normal qu'entre les joueurs sélectionnés sur leur position et les position finales gardées il y a une différence car certaines sont de corrélation 1 """
#J'ai enlevé un maximum de position car en terme de prédiction on est à une corrélation de 1

#Cette partie de code n'est plus fonctionelle car ces variables ont étés supprimées, elle sert d'illustration des traitement des données
#J'ai enlevé tous les postes car trop grande note par rapport aux autres variables
#J'ai enlevé ensuite valeur du joueur pour la même raison

#colonnes_a_modifier = ['lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm','rm', 'ls', 'st', 'rs','lw', 'lf', 'cf', 'rf','rw']
# colonnes_a_modifier = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lcm', 'cm', 'rcm']

# for colonne in colonnes_a_modifier:
    
#     data_ST[colonne] = data_ST[colonne].str.replace('+3', '')
#     data_ST[colonne] = data_ST[colonne].str.replace('+2', '')
#     data_ST[colonne] = data_ST[colonne].str.replace('+1', '')
#     data_ST[colonne] = data_ST[colonne].str.replace('-2', '')
#     data_ST[colonne] = data_ST[colonne].str.replace('-1', '')
#     data_ST[colonne] = data_ST[colonne].str.replace('-3', '')
#     data_ST[colonne] = data_ST[colonne].str.replace('-4', '')

# # Les collones sont converties en valeur numérique
# data_ST[colonnes_a_modifier] = data_ST[colonnes_a_modifier].apply(pd.to_numeric)

""" J'ai supprimé les joueurs dont la value_eur était égale à 0 ou bien il n'y avait pas de données. Après avoir regardé de près ces joueurs ce sont
soit des joueurs très âgé ou bien des joueurs pour lesquels il manque beaucoup d'autres données.  """

#La variables 'fifa_update' est suppprimée car elle servait uniquement de sélection de données les plus récentes
ST_final = data_ST.drop(columns=['fifa_update']).dropna(subset=['value_eur']) #La fonction dropna supprime les lignes pour lesquelles il n'y a pas de valeur pour 'value_eur'

ST_final.reset_index(drop=True, inplace=True) # réinitialiser l'index après la suppression des lignes

# Les données de la colonne 'work_rate' seront réparties dans deux colonnes distinctes.
ST_final[['Attacking_work_rate', 'Defensive_work_rate']] = ST_final['work_rate'].str.split('/', expand=True) #Je crée 2 nouvelles colonnes
ST_final.drop('work_rate', axis=1, inplace=True)

# Je veux changer mes chaînes de caractères en valeurs numériques
mapping = {'Low': 1, 'Medium': 2, 'High': 3}

# Remplacer les valeurs par des valeurs numériques
ST_final['Attacking_work_rate'] = ST_final['Attacking_work_rate'].replace(mapping)
ST_final['Defensive_work_rate'] = ST_final['Defensive_work_rate'].replace(mapping)

foot = {'Left': 0, 'Right': 1}

# Transformer les valeurs dans la colonne 'preferred_foot' en entiers
ST_final['preferred_foot'] = ST_final['preferred_foot'].map(foot)

print("Composition de ST_final:", ST_final.shape)

# #Je m'assure qu'il n'y a pas de données manquantes
# données_manquantes = ST_final.columns[ST_final.isna().any()]
# if données_manquantes.empty:
#     print("Il n'y a pas de valeurs manquantes dans votre base de données.")
# else:
#     print("Colonnes avec des valeurs manquantes:")
#     print(ST_final[données_manquantes].isna().sum())



""" Analyse de corrélation entre overall et les variables dépendantes """
# correlation_matrix = ST_final.corr()

# high_corr_matrix = correlation_matrix[(correlation_matrix > 0.75) | (correlation_matrix < -0.75)]

# plt.figure(figsize=(10, 8))

# plt.imshow(high_corr_matrix, cmap='coolwarm', interpolation='nearest')

# plt.colorbar()

# plt.title('Matrice de Corrélation')
# plt.xticks(np.arange(len(high_corr_matrix.columns)), high_corr_matrix.columns, rotation=90)
# plt.yticks(np.arange(len(high_corr_matrix.columns)), high_corr_matrix.columns)

# plt.show()

# # Donne un aperçu des variables restantes et de leurs valeurs.
# first_row = ST_final.iloc[0]  # Sélection de la première ligne
# column_names = ST_final.columns  # Obtention des noms des colonnes
# for column_name, value in zip(column_names, first_row):
#     print(f"{column_name}: {value}")

# Après avoir analysé la matrice de corrélation, je supprime les variables corrélées.
X_overall = ST_final.drop(["overall","movement_reactions","skill_ball_control", "mentality_positioning","attacking_finishing","potential",
                           "attacking_short_passing","attacking_volleys","skill_dribbling","power_shot_power","power_long_shots", "pace",
                           "mentality_composure","passing","shooting","dribbling","defending","defending_sliding_tackle", "mentality_vision"], axis=1)
              

y_overall = ST_final["overall"]

#Je sépare mes données en donnée d'entrainement et de test
X_train_overall, X_test_overall, y_train_overall, y_test_overall = train_test_split(X_overall,y_overall, random_state=42, test_size=0.3)




""" Ridge regression """
# Mise en place du modèle avec les pénalités alphas
model = RidgeCV(cv=None, alphas= [0.3 ,0.4 ,0.5 ,0.6 , 0.7],store_cv_values=True)

# 3 méthodes de mise à l'échelle de données
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()

# Mes variables sont mises à l'échelle
X_train_Scaled = scaler.fit_transform(X_train_overall)
X_test_Scaled = scaler.transform(X_test_overall)

# Le modèle s'entraine sur les données
model.fit(X_train_Scaled, y_train_overall)

# Utilisation de la validation croisée
cv_scores = cross_val_score(model, X_train_Scaled, y_train_overall, cv=5)
print("Scores de validation croisée :", cv_scores)

best_alpha = model.alpha_
print("Best alpha:", best_alpha)

# Je vais afficher ici les coefficients des variables 
coefficients_with_names = dict(zip(X_train_overall.columns, model.coef_))

sorted_coefficients = sorted(coefficients_with_names.items(), key=lambda x: abs(x[1]), reverse=True)

top_coefficients = sorted_coefficients[:30]

print("Top 15 coefficients with highest values:")
for feature_name, coefficient in top_coefficients:
    print(f"{feature_name}: {coefficient}")

y_pred = model.predict(X_test_Scaled)

# Dans la fonction RidgeCV() cv_values renvoient le MSE
cv_mse = np.mean(model.cv_values_, axis=0)

print("Scores de validation croisée moyens pour chaque valeur d'alpha:")
for alpha, score in zip(model.alphas, cv_mse):
    print(f"Alpha = {alpha}: {score}")

""" Mesure de performance """

mae = mean_absolute_error(y_test_overall, y_pred)
mse = mean_squared_error(y_test_overall, y_pred)
score = model.score(X_test_Scaled, y_test_overall)
score = model.score(y_test_overall, y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("MSE:", mse)
print("Score R^2 sur les données de test :", score)
r2 = r2_score(y_test_overall, y_pred)
print("R2 Score:", r2)

def calculate_aic(n, rss, k):
    aic = n * np.log(rss / n) + 2 * k
    return aic

def calculate_bic(n, rss, k):
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


n = X_test_Scaled.shape[0]

rss = np.sum((y_test_overall - y_pred) ** 2)

k = np.sum(model.coef_ != 0)

aic = calculate_aic(n, rss, k)
print("Akaike Information Criterion (AIC):", aic)

bic = calculate_bic(n, rss, k)
print("Bayesian Information Criterion (BIC):", bic)

r2_ajuste = 1-(1-r2)*(n-1)/(n-k-1)
print("La valeur du r2 ajusté :", r2_ajuste)


# #test de significativité statistique
# n_bootstraps = 1000
# n_size = int(len(X_train_Scaled) * 0.8)
# bootstrapped_coefs = np.zeros((n_bootstraps, X_train_Scaled.shape[1]))

# for i in range(n_bootstraps):
#     X_resampled, y_resampled = resample(X_train_Scaled, y_train_overall, n_samples=n_size)
    
#     model.fit(X_resampled, y_resampled)
    
#     bootstrapped_coefs[i] = model.coef_

# # Calcul des intervalles de confiance à 95%
# confidence_intervals = np.percentile(bootstrapped_coefs, [2.5, 97.5], axis=0)

# # Vérification de la significativité des coefficients
# significance = (confidence_intervals[0] > 0) | (confidence_intervals[1] < 0)

# print("Intervalles de confiance à 95% et significativité des coefficients:")
# for i, (feature_name, coefficient) in enumerate(coefficients_with_names.items()):
#     ci_lower, ci_upper = confidence_intervals[:, i]
#     is_significant = significance[i]
#     print(f"{feature_name}: Coef = {coefficient}, CI = [{ci_lower}, {ci_upper}], Significatif: {is_significant}")



""" Analyse graphique """
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve


# train_sizes, train_scores, valid_scores = learning_curve(model, X_train_Scaled, y_train_overall, train_sizes=np.linspace(0.1, 1.0, 10), cv=15, scoring='neg_mean_squared_error')

# train_errors, valid_errors = -train_scores.mean(axis=1), -valid_scores.mean(axis=1)


# plt.figure()
# plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, valid_errors, 'o-', color="g", label="Cross-validation error")
# plt.xlabel("Training examples")
# plt.ylabel("Mean Squared Error")
# plt.title("Learning Curve")
# plt.legend(loc="best")
# plt.show()


# # Calculer les résidus
# residuals = y_test_overall - y_pred

# # Tracer les résidus..
# plt.figure(figsize=(8, 6))
# plt.scatter(y_pred, residuals, color='blue')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title('Analyse des résidus')
# plt.xlabel('Prédictions')
# plt.ylabel('Résidus')
# plt.grid(True)
# plt.show()

""" Lasso regression """

# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()


# X_train_scaled = scaler.fit_transform(X_train_overall)
# X_test_scaled = scaler.transform(X_test_overall)

# #Fonction de Lasso avec les différentes pénalités 
# model = LassoCV(cv=5, alphas= [ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fit_intercept=False)

# #Entrainement du modèle
# model.fit(X_train_scaled, y_train_overall)

# y_pred = model.predict(X_test_scaled)

# #Analyse du MSE en fonction des différents alphas 
# mean_cv_score = np.mean(model.mse_path_, axis=0)  
# print("Scores de validation croisée moyens pour chaque valeur d'alpha:")
# for alpha, score in zip(model.alphas_, mean_cv_score):
#     print(f"Alpha = {alpha}: {score}")


# coefficients_with_names = dict(zip(X_train_overall.columns, model.coef_))

# selected_features = {feature: coef for feature, coef in coefficients_with_names.items() if coef != 0}
# selected_features_sorted = dict(sorted(selected_features.items(), key=lambda x: abs(x[1]), reverse=True))

# print("Variables sélectionnées avec leurs coefficients :", selected_features_sorted)

# print("Meilleur alpha :", model.alpha_)


# """ Analyse des résultats """
# score = model.score(X_test_scaled, y_test_overall)
# print("Score R^2 sur les données de test :", score)

# r2 = r2_score(y_test_overall, y_pred)
# print("R2 Score:", r2)


# mae = mean_absolute_error(y_test_overall, y_pred)
# print("Mean Absolute Error:", mae)

# mse = mean_squared_error(y_test_overall, y_pred)
# print("MSE:", mse)

# rmse = np.sqrt(mean_squared_error(y_test_overall, y_pred))
# print("Root Mean Squared Error:", rmse)


# def calculate_aic(n, rss, k):
#     aic = n * np.log(rss / n) + 2 * k
#     return aic


# def calculate_bic(n, rss, k):
#     bic = n * np.log(rss / n) + k * np.log(n)
#     return bic

# n = X_test_scaled.shape[0]

# rss = np.sum((y_test_overall - y_pred) ** 2)

# k = np.sum(model.coef_ != 0)

# aic = calculate_aic(n, rss, k)
# print("Akaike Information Criterion (AIC):", aic)

# bic = calculate_bic(n, rss, k)
# print("Bayesian Information Criterion (BIC):", bic)

# r2_ajuste = 1-(1-r2)*(n-1)/(n-k-1)
# print("La valeur du r2 ajusté :", r2_ajuste)



# """ Analyse graphique """

# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve


# # train_sizes, train_scores, valid_scores = learning_curve(model, X_train_scaled, y_train_overall, train_sizes=np.linspace(0.1, 1.0, 10), cv=15, scoring='r2')
# train_sizes, train_scores, valid_scores = learning_curve(model, X_train_scaled, y_train_overall, train_sizes=np.linspace(0.1, 1.0, 10), cv=15, scoring='neg_mean_squared_error')
# train_errors, valid_errors = -train_scores.mean(axis=1), -valid_scores.mean(axis=1)


# plt.figure()
# plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, valid_errors, 'o-', color="g", label="Cross-validation error")
# plt.xlabel("Training examples")
# plt.ylabel("MSE")
# plt.title("Learning Curve : Lasso regression")
# plt.legend(loc="best")
# plt.show()

