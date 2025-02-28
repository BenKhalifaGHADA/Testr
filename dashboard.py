#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install dash


# In[13]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
from dash.dependencies import Input, Output


# In[14]:


# Charger le dataset
df = pd.read_csv('student_assessment_data.csv')


# In[15]:


# Calculer les statistiques
average_score = df['quiz_score'].mean()
completion_rate = df['completion_status'].value_counts(normalize=True) * 100
average_participation = df['participation_score'].mean()
average_content_quality = df['content_quality_rating'].mean()
average_engagement = df['engagement_score'].mean()
pass_rate = (df['quiz_score'] >= 60).mean() * 100  # Supposant que 60 est la note de passage

# Sélectionner les caractéristiques pour le clustering
features = df[['quiz_score', 'engagement_score', 'participation_score']]

# Standardiser les caractéristiques
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Déterminer le nombre optimal de clusters en utilisant la méthode du coude
inertia = []
K = range(1, 6)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Choisir le nombre optimal de clusters (par exemple, 3 basé sur la méthode du coude)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Générer des feedbacks automatisés basés sur les clusters
def generate_feedback2(cluster):
    if cluster == 0:
        return "Excellent performance! Keep up the great work and continue to engage actively."
    elif cluster == 1:
        return "Good job! You are performing well, but consider increasing your engagement in class activities."
    elif cluster == 2:
        return "It seems you may need additional support. Please reach out for help and consider focusing on your participation."

# Appliquer la génération de feedback
df['feedback'] = df['cluster'].apply(generate_feedback2)

# Créer l'application Dash
app = Dash()

app.layout = html.Div([
    html.H1(children='Tableau de bord personnalisé des étudiants', style={'textAlign': 'center'}),
    dcc.Dropdown(df.student_name.unique(), 'Alice', id='dropdown-selection'),
    html.Div(id='metrics-content', style={'textAlign': 'center', 'marginTop': '20px'}),
    dcc.Graph(id='graph-content')
])

@app.callback(
    [Output('metrics-content', 'children'),
     Output('graph-content', 'figure')],
    [Input('dropdown-selection', 'value')]
)
def update_dashboard(value):
    dff = df[df.student_name == value]
    metrics = [
        html.H4(children='Métriques pour l\'étudiant sélectionné'),
        html.P(f'Nom : {dff["student_name"].values[0]}'),
        html.P(f'Score du quiz : {dff["quiz_score"].values[0]}'),
        html.P(f'Score d\'engagement : {dff["engagement_score"].values[0]}'),
        html.P(f'Score de participation : {dff["participation_score"].values[0]}'),
        html.P(f'Évaluation de la qualité du contenu : {dff["content_quality_rating"].values[0]}'),
        html.P(f'Taux de complétion : {completion_rate[dff["completion_status"].values[0]]:.2f}%'),
        html.P(f'Feedback : {dff["feedback"].values[0]}')
    ]
    fig = go.Figure(data=[
        go.Bar(name='Quiz Score', x=dff['student_name'], y=dff['quiz_score']),
        go.Bar(name='Engagement Score', x=dff['student_name'], y=dff['engagement_score']),
        go.Bar(name='Participation Score', x=dff['student_name'], y=dff['participation_score'])
    ])
    fig.update_layout(barmode='group', title='Scores des étudiants')
    return metrics, fig

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:




