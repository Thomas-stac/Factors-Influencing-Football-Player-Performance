import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve


ST = pd.read_csv('Base_de_données.csv')

#Affiche le nombre de lignes et colonnes
print("Base de données des joueurs offensifs de 2023:", ST.shape)



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
#Je vais transformer mes donées string en int
# ST_final['club_position'] = ST_final['club_position'].replace('RES', 'SUB')

X_overall = ST_final.drop(["overall","value_eur", "wage_eur","movement_reactions","skill_ball_control", "mentality_positioning","attacking_finishing",
                           "attacking_short_passing","attacking_volleys","skill_dribbling","power_shot_power","power_long_shots","mentality_composure",
                           "passing","shooting","dribbling","defending","defending_sliding_tackle","mentality_vision","pace", "potential"], axis=1)

y_overall = ST_final["overall"]

X_train_overall, X_test_overall, y_train_overall, y_test_overall = train_test_split(X_overall,y_overall, random_state=42, test_size=0.3)

""" correlation_matrix = X_overall.corr()

high_corr_matrix = correlation_matrix[(correlation_matrix > 0.75) | (correlation_matrix < -0.75)]

# Création de la figure
plt.figure(figsize=(10, 8))

# Tracé du tableau de corrélation
plt.imshow(high_corr_matrix, cmap='coolwarm', interpolation='nearest')

# Ajout d'une barre de couleur
plt.colorbar()

# Ajout des titres et des étiquettes d'axe
plt.title('Matrice de Corrélation')
plt.xticks(np.arange(len(high_corr_matrix.columns)), high_corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(high_corr_matrix.columns)), high_corr_matrix.columns)

# Affichage du tableau
plt.show()


first_row = ST_final.iloc[0]  # Sélection de la première ligne
column_names = ST_final.columns  # Obtention des noms des colonnes
for column_name, value in zip(column_names, first_row):
    print(f"{column_name}: {value}") """

""" Biais de représentativité pour l'attribut valeur_eur """
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='overall', y='value_eur', data=ST_final)
# plt.title('Distribution du salaire en euros en fonction de Overall')
# plt.xlabel('Overall')
# plt.ylabel('salaire en euros')
# plt.grid(True)
# plt.show()

""" Implémentation Random Forest Regressor """
""" n_estimators est égal au nombre d'arbre dans le random forest """
#J'ai enlevé tous les postes car trop grande note par rapport aux autres variables
#J'ai enlevé ensuite valeur du joueur pour la même raison

#Je présente les hyperparamètres et leur différentes valeurs. Grace au GridSearchCV toute les combinaisons vont être testées
param_grid = {
    'n_estimators': [30, 40, 50],  # Nombre d'arbres dans la forêt
    'max_depth': [20, 30],       # Profondeur maximale de chaque arbre
    'min_samples_split': [60, 100],   # Nombre minimal d'échantillons requis pour diviser un nœud
    'min_samples_leaf': [50, 60]      # Nombre minimal d'échantillons requis pour être une feuille
}

rf_regressor = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train_overall, y_train_overall)

#C'est mon RFR avec les meilleurs hyperparamètres
y_pred = grid_search.best_estimator_.predict(X_test_overall)

r2 = r2_score(y_test_overall, y_pred)
print("R-squared:", r2)

# Affichee les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres:", grid_search.best_params_)

# Accès à l'importance des attributs
importances = grid_search.best_estimator_.feature_importances_

# Création d'un DataFrame pour stocker les noms des attributs et leurs importances
features = pd.DataFrame({'Feature': X_train_overall.columns, 'Importance': importances})

# Triez les attributs par importance décroissante 
top_features = features.sort_values(by='Importance', ascending=False).head(15)
print(top_features)

""" Mesures de performances """

mse = mean_squared_error(y_test_overall, y_pred)

print("MSE:", mse)

mae = mean_absolute_error(y_test_overall, y_pred)

rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)


def calculate_aic(n, rss, k):
    aic = n * np.log(rss / n) + 2 * k
    return aic


def calculate_bic(n, rss, k):
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic

# Nombre d'observations
n = X_test_overall.shape[0]

# Résidus de la somme des carrés (RSS)
rss = np.sum((y_test_overall - y_pred) ** 2)

# Nombre de paramètres estimés
k = len(grid_search.best_estimator_.feature_importances_)

# Calcul de l'AIC
aic = calculate_aic(n, rss, k)
print("Akaike Information Criterion (AIC):", aic)


bic = calculate_bic(n, rss, k)
print("Bayesian Information Criterion (BIC):", bic)

r2_ajuste = 1-(1-r2)*(n-1)/(n-k-1)
print("La valeur du r2 ajusté :", r2_ajuste)


#Représentation graphique
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Utilisation de la fonction pour tracer la courbe d'apprentissage
title = "Learning Curves (RFR)"
plot_learning_curve(grid_search.best_estimator_, title, X_train_overall, y_train_overall, cv=5, n_jobs=-1)

plt.show()


# def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5):
#     train_scores, test_scores = validation_curve(
#         estimator, X, y, param_name=param_name, param_range=param_range,
#         cv=cv, scoring="r2", n_jobs=-1)
    
#     train_scores_mean = np.mean(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(param_range, train_scores_mean, label="Training score")
#     plt.plot(param_range, test_scores_mean, label="Cross-validation score")
#     plt.title("Validation Curve")
#     plt.xlabel(param_name)
#     plt.ylabel("R-squared")
#     plt.legend(loc="best")
#     plt.grid()
#     plt.show()

# # Paramètres pour la courbe de validation
# param_range = [10, 20, 30]
# param_name = 'n_estimators'

# # Tracer la courbe de validation
# plot_validation_curve(grid_search.best_estimator_, X_train_overall, y_train_overall, param_name, param_range)

