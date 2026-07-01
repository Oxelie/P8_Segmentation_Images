### Étape 1 : Installer Streamlit

Si vous ne l'avez pas encore fait, installez Streamlit en utilisant pip :

```bash
pip install streamlit
```

### Étape 2 : Créer l'application Streamlit

Créez un fichier Python pour votre application Streamlit, par exemple `app.py`. Voici un exemple de code qui consomme une API Flask :

```python
import streamlit as st
import requests

# URL de votre API Flask déployée sur Heroku
API_URL = "https://votre-api.herokuapp.com/endpoint"

def main():
    st.title("Application Streamlit avec API Flask")

    # Créer un formulaire pour l'entrée utilisateur
    with st.form(key='my_form'):
        user_input = st.text_input("Entrez quelque chose :")
        submit_button = st.form_submit_button(label='Soumettre')

    if submit_button:
        # Appeler l'API Flask avec l'entrée utilisateur
        response = requests.post(API_URL, json={"input": user_input})

        if response.status_code == 200:
            # Traiter la réponse de l'API
            data = response.json()
            st.success(f"Réponse de l'API : {data['result']}")
        else:
            st.error("Erreur lors de l'appel à l'API.")

if __name__ == "__main__":
    main()
```

### Étape 3 : Exécuter l'application Streamlit

Pour exécuter votre application Streamlit, utilisez la commande suivante dans votre terminal :

```bash
streamlit run app.py
```

### Étape 4 : Tester l'application

Ouvrez votre navigateur et allez à l'adresse indiquée dans le terminal (généralement `http://localhost:8501`). Vous devriez voir votre application Streamlit. Entrez des données dans le formulaire et soumettez-les pour voir la réponse de votre API Flask.

### Étape 5 : Déployer l'application Streamlit (facultatif)

Si vous souhaitez déployer votre application Streamlit, vous pouvez utiliser des services comme Streamlit Sharing, Heroku, ou d'autres plateformes cloud. Voici un exemple de déploiement sur Heroku :

1. Créez un fichier `requirements.txt` pour spécifier les dépendances :

```
streamlit
requests
```

2. Créez un fichier `Procfile` pour indiquer à Heroku comment exécuter votre application :

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

3. Initialisez un dépôt Git, ajoutez vos fichiers, et déployez sur Heroku :

```bash
git init
git add .
git commit -m "Initial commit"
heroku create votre-nom-d-app
git push heroku master
```

### Conclusion

Vous avez maintenant une application Streamlit qui consomme une API Flask déployée sur Heroku. Vous pouvez personnaliser l'interface utilisateur et les fonctionnalités selon vos besoins. N'hésitez pas à poser des questions si vous avez besoin de plus d'aide !