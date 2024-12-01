from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


ST = pd.read_csv('Base_de_données.csv')

#Affiche le nombre de lignes et colonnes
print("Base de données des joueurs offensifs de 2023:", ST.shape)

#J'ai fait varier mes suppressions de variables afin de voir si l'impact de certaines n'étaient pas contre intuitive comme pour 'club_jersey_number',
#néanmoins elles n'ont jamais eu d'impact dans mes modèles.


# Hypothèses :
colonnes_a_supprimer = ['player_url' ,'fifa_update_date','long_name','short_name', 'league_name', 'club_name', 'club_jersey_number', 'fifa_version',
                        'nationality_name', 'nation_jersey_number', 'player_face_url', 'club_joined_date','club_contract_valid_until_year',
                        'dob', 'club_loaned_from', 'nation_team_id', 'nation_position', 'player_tags', 'player_traits', 'goalkeeping_speed', 
                        'release_clause_eur', 'player_positions', 'real_face', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                        'goalkeeping_positioning', 'goalkeeping_reflexes', 'lwb','ldm','cdm','rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                        'lm', 'lcm', 'cm', 'rcm','rm','lw', 'rf','lf','rs','ls','rw', 'lam', 'cam', 'st','cf', 'ram', 'club_position', 'body_type',
                        'player_id'
                        ]
data_ST = ST.drop(columns=colonnes_a_supprimer)


""" C'est normal qu'entre les joueurs sélectionné sur leur position et les position finales gardées il y a une différence car certaines
sont de corrélation 1 """


# ST_final = data_ST.dropna(subset=['value_eur'])
ST_final = data_ST.drop(columns=['fifa_update']).dropna(subset=['value_eur'])

ST_final.reset_index(drop=True, inplace=True)

ST_final[['Attacking_work_rate', 'Defensive_work_rate']] = ST_final['work_rate'].str.split('/', expand=True)
ST_final.drop('work_rate', axis=1, inplace=True)

# Mapping des valeurs
mapping = {'Low': 1, 'Medium': 2, 'High': 3}

# Remplacer les valeurs dans les nouvelles colonnes
ST_final['Attacking_work_rate'] = ST_final['Attacking_work_rate'].replace(mapping)
ST_final['Defensive_work_rate'] = ST_final['Defensive_work_rate'].replace(mapping)

foot = {'Left': 0, 'Right': 1}

# Transformer les valeurs dans la colonne 'preferred_foot' en entiers
ST_final['preferred_foot'] = ST_final['preferred_foot'].map(foot)

print("Composition de ST_final:", ST_final.shape)
#J'ai enlevé 'club_position' du get_dummies et je l'ai rajouté dans la collone_supprimer

""" Biais de représentativité pour l'attribut valeur_eur """
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='overall', y='wage_eur', data=ST_final)
# plt.title('Distribution du salaire en euros en fonction de Overall')
# plt.xlabel('Overall')
# plt.ylabel('salaire en euros')
# plt.grid(True)
# plt.show()

# Après avoir analysé la matrice de corrélation, je supprime les variables corrélées.
X_overall = ST_final.drop(["overall","movement_reactions","skill_ball_control", "mentality_positioning","attacking_finishing", "potential",
                           "attacking_short_passing","attacking_volleys","skill_dribbling","power_shot_power","power_long_shots", "pace",
                           "mentality_composure","passing","shooting","dribbling","defending","defending_sliding_tackle", "mentality_vision"], axis=1)
y_overall = ST_final["overall"]


# #Mise à l'échelle des données
# scaler = StandardScaler()
# # scaler = RobustScaler()
# # scaler = MinMaxScaler()

# X_overall_scaled = scaler.fit_transform(X_overall)

# X_train_overall, X_test_overall, y_train_overall, y_test_overall = train_test_split(X_overall_scaled, y_overall, random_state=42, test_size=0.3)

# """ Je commence mon recursive feature elimination. """

# param_grid = {
#     'n_estimators': [5, 9],  # Nombre d'arbres dans la forêt
#     'max_depth': [10, 20],       # Profondeur maximale de chaque arbre
#     'min_samples_split': [60, 100,],   # Nombre minimal d'échantillons requis pour diviser un nœud
#     'min_samples_leaf': [40, 50]       # Nombre minimal d'échantillons requis pour être une feuille
# }

# #Mon modèle de sélection c'est le Random Forest Regressor
# rf_regressor = RandomForestRegressor(random_state=42)

# grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, n_jobs=-1)

# grid_search.fit(X_train_overall, y_train_overall)

# # Afficher les meilleurs hyperparamètres trouvés
# print("Meilleurs hyperparamètres:", grid_search.best_params_)

# # Je lance mon Random Forest Regressor (RFE) pour sélectionner X variables
# selector = RFE(rf_regressor, n_features_to_select=10, step=3)  #Step correspond au nombre de variables enlevées à chaque étape
# selector = selector.fit(X_train_overall, y_train_overall)

# #On cherche à garder les variable de ranking 1
# ranking = selector.ranking_

# # Cet attribut est un tableau booléen qui indique quelles caractéristiques ont été sélectionnées (True) et quelles caractéristiques ont été éliminées (False) par le processus de sélection des caractéristiques
# selected_features = selector.support_

# X_overall_df = pd.DataFrame(X_overall, columns=X_overall.columns)

# #On regarde les variables en fonction des leur ranking. On sélectione les variable avec un ranking de 1.
# features = pd.DataFrame({'Feature': X_overall_df.columns, 'Importance': ranking, 'Selected': selected_features})


# important_features = features[features['Selected'] == True].sort_values('Importance')

# selected_features_importance = selector.estimator_.feature_importances_


# selected_features_names = features[features['Selected'] == True]['Feature'].values

# #Je veux voir toutes mes variables avec leur ranking et importance
# for i, feature_name in enumerate(selected_features_names):
#     if ranking[i] == 1:
#         print(f"Feature {feature_name}: Ranking = {ranking[i]}, Importance = {selected_features_importance[i]}")


""" Implémentation du KNN """

# columns = ST_final.drop(["overall","shooting","passing","dribbling","defending"], axis=1).columns

#Je prends les variables sélectionnées plus haut.
# X_train_selected = pd.DataFrame(X_train_overall, columns=X_overall.columns)[selected_features_names]
# X_test_selected = pd.DataFrame(X_test_overall, columns=X_overall.columns)[selected_features_names]


# knn = KNeighborsRegressor()

# param_grid = {
#     'n_neighbors': [ 10, 15], #Nombre de voisins 
#     'algorithm': ['auto' ,'ball_tree', 'kd_tree', 'brute'], #Fonction qui va déterminer le nombre optimal de données à garder en noeuds final
#     'leaf_size': [130, 150], #Nombre de données dans le noeuds final
# }

# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')

# grid_search.fit(X_train_selected, y_train_overall)

# best_model = grid_search.best_estimator_

# print("Meilleurs paramètres:", grid_search.best_params_)

# y_pred = best_model.predict(X_test_selected)

# mse = mean_squared_error(y_test_overall, y_pred)

# print("MSE:", mse)


# best_model.fit(X_train_selected, y_train_overall)

# #On regarde comment le modèle se comporte sur des données test qu'il n'a pas encore vu
# predictions = best_model.predict(X_test_selected)

# """ Mesures de performances """

# mae = mean_absolute_error(y_test_overall, predictions)

# rmse = np.sqrt(mse)
# print("Mean Absolute Error:", mae)
# print("Root Mean Squared Error:", rmse)

# r2 = r2_score(y_test_overall, predictions)
# print("R2 Score:", r2)

# def calculate_aic(n, rss, k):
#     aic = n * np.log(rss / n) + 2 * k
#     return aic

# def calculate_bic(n, rss, k):
#     bic = n * np.log(rss / n) + k * np.log(n)
#     return bic

# n = X_test_selected.shape[0]
# rss = np.sum((y_test_overall - y_pred) ** 2)
# k = len(selected_features_names)  # Nombre de variables sélectionnées

# aic = calculate_aic(n, rss, k)
# print("Akaike Information Criterion (AIC):", aic)

# bic = calculate_bic(n, rss, k)
# print("Bayesian Information Criterion (BIC):", bic)

# r2_ajuste = 1 - (1 - r2) * (n - 1) / (n - k - 1)
# print("La valeur du r2 ajusté :", r2_ajuste)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2')
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# title = "Learning Curves (KNN)"
# plot_learning_curve(best_model, title, X_train_selected, y_train_overall, cv=5, n_jobs=-1)

# plt.show()


""" Implémentation KNN avec lasso penalisation """
X_train_overall, X_test_overall, y_train_overall, y_test_overall = train_test_split(X_overall, y_overall, random_state=42, test_size=0.3)

scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_overall)
X_test_scaled = scaler.transform(X_test_overall)

""" Mise en place du modèle de Lasso pour la sélection de variables """
model = LassoCV(cv=5, alphas= [0.7, 0.75], fit_intercept=True)

#j'ai remplacé overall par scaled
model.fit(X_train_overall, y_train_overall)

#J'analyse le MSE pour chaque alphas testé
mean_cv_score = np.mean(model.mse_path_, axis=0)  
print("Scores de validation croisée moyens pour chaque valeur d'alpha:")
for alpha, score in zip(model.alphas_, mean_cv_score):
    print(f"Alpha = {alpha}: {score}")

column_names = X_overall.columns.tolist()
coefficients_with_names = dict(zip(column_names, model.coef_))

#Je répertorie mes variables sélectionnées avec leur coefficients qui sont triés par ordre décroissant
selected_features = {feature: coef for feature, coef in coefficients_with_names.items() if coef != 0}
selected_features_names = {feature for feature, coef in coefficients_with_names.items() if coef != 0}
sorted_features = sorted(selected_features.items(), key=lambda x: x[1], reverse=True)

print("Variables sélectionnées avec leurs coefficients :", sorted_features)

print("Meilleur alpha :", model.alpha_)

score = model.score(X_test_overall, y_test_overall)
print("Score R^2 Lasso sur les données de test :", score)


""" Mise en place du KNeighborsRegressor """     
from sklearn.neighbors import KNeighborsRegressor

#Je travaille sur base des données sélectionnées précédemment
selected_features_names = list(selected_features_names)

X_train_scaled = pd.DataFrame(X_train_overall, columns=X_overall.columns)

X_train_selected = X_train_scaled[selected_features_names]

X_test_scaled = pd.DataFrame(X_test_overall, columns=X_overall.columns)
X_test_selected = X_test_scaled[selected_features_names]

knn = KNeighborsRegressor()

#On va utiliser la fonction GridSearchCV afin de trouver le meilleur ensemble d'hyperparamètres
param_grid = {
    'n_neighbors': [ 6, 8, 10, 12, 14],  # ce paramètre va le plus jouer sur ma learning curve
    'algorithm': [ 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30]
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train_selected, y_train_overall)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_selected)

print("repsectivement  ypred :", y_pred.shape )
print("repsectivement y test overall  :", y_test_overall.shape )
mse = mean_squared_error(y_test_overall, y_pred)



print("MSE:", mse)
print("Meilleurs paramètres:", grid_search.best_params_)

# Je prends le KNN avec les meilleurs hyperparamètres
knn_regressor = best_model

knn_regressor.fit(X_train_selected, y_train_overall)

# Le modèle est lancé sur des données test pour voir sa performance sur de nouvelles données
y_pred = knn_regressor.predict(X_test_selected)


""" Analyse des mesures de performance """
mse = mean_squared_error(y_test_overall, y_pred)

print(f'Mean Squared Error: {mse}')


mae = mean_absolute_error(y_test_overall, y_pred)
print("Mean Absolute Error:", mae)


rmse = np.sqrt(mean_squared_error(y_test_overall, y_pred))
print("Root Mean Squared Error:", rmse)


r2 = r2_score(y_test_overall, y_pred)
print("R2 Score:", r2)

def calculate_aic(n, rss, k):
    aic = n * np.log(rss / n) + 2 * k
    return aic

def calculate_bic(n, rss, k):
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


n = X_test_selected.shape[0]
rss = np.sum((y_test_overall - y_pred) ** 2)
k = len(selected_features_names)

aic = calculate_aic(n, rss, k)
print("Akaike Information Criterion (AIC):", aic)

bic = calculate_bic(n, rss, k)
print("Bayesian Information Criterion (BIC):", bic)

r2_ajuste = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print("La valeur du r2 ajusté :", r2_ajuste)


# Analyse graphique
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


train_sizes, train_scores, valid_scores = learning_curve(best_model, X_train_selected, y_train_overall, train_sizes=np.linspace(0.1, 1.0, 10), cv=15, scoring='neg_mean_squared_error')

train_errors, valid_errors = -train_scores.mean(axis=1), -valid_scores.mean(axis=1)


plt.figure()
plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
plt.plot(train_sizes, valid_errors, 'o-', color="g", label="Cross-validation error")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve : KNN - Lasso")
plt.legend(loc="best")
plt.show()



# # # Calculer les résidus
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



""" R2 en fonction de K """

# # Plage de valeur de K à tester
# neighbors = range(1,20)

# # Liste pour stocker les scores R^2 pour chaque valeur de K
# r2_scores = []

# for k in neighbors:
#     knn = KNeighborsRegressor(n_neighbors=k)
#     knn.fit(X_train_selected, y_train_overall)
    
#     r2 = knn.score(X_test_selected, y_test_overall)
    
#     r2_scores.append(r2)

# # Tracé de l'évolution du R^2 en fonction du nombre de voisins
# plt.plot(neighbors, r2_scores, marker='o')
# plt.xlabel('Nombre de voisins (K)')
# plt.ylabel('R2 Score')
# plt.title('Evolution du R2 en fonction de K')
# plt.grid(True)
# plt.show()






