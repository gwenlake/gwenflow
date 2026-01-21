import requests
import json
import os
from gwenflow import ChatOpenAI
from gwenflow.agents.agent import Agent
from gwenflow.tools import FunctionTool

os.environ["OPENAI_API_KEY"] = "sk-proj-................................" 

OMDB_API_KEY = "OMDB_API_KEY"

def search_best_books(topic: str) -> str:
    """Cherche les livres les plus pertinents sur un thème."""
    url = f"https://www.googleapis.com/books/v1/volumes?q={topic}&orderBy=relevance&maxResults=3"
    try:
        data = requests.get(url).json()
        if "items" not in data: return "Rien trouvé."
        books = []
        for item in data["items"]:
            info = item.get("volumeInfo", {})
            books.append(f"- Livre : {info.get('title')} ({info.get('authors', ['?'])[0]})")
        return "\n".join(books)
    except Exception as e: return str(e)

def search_cult_movies(topic: str) -> str:
    """Cherche des films populaires sur un thème."""
    url = f"http://www.omdbapi.com/?s={topic}&type=movie&apikey={OMDB_API_KEY}"
    try:
        data = requests.get(url).json()
        if data.get("Response") == "False": return "Rien trouvé."
        movies = []
        # On prend les 3 premiers résultats
        for item in data.get("Search", [])[:3]:
            movies.append(f"- Film : {item.get('Title')} ({item.get('Year')})")
        return "\n".join(movies)
    except Exception as e: return str(e)

tool_books = FunctionTool.from_function(search_best_books)
tool_movies = FunctionTool.from_function(search_cult_movies)

llm = ChatOpenAI(model="gpt-4o-mini")

agent_libraire = Agent(
    name="Libraire",
    instructions=[
        "Tu es un bibliothécaire érudit spécialisé dans les ouvrages de référence.",
        "Ton objectif est de trouver les livres les plus pertinents, fondateurs ou acclamés par la critique sur le sujet donné.",
        "Ne te contente pas de n'importe quel livre : cherche les auteurs qui font autorité ou les romans cultes.",
        "Utilise ton outil 'search_best_books' pour valider les titres et les auteurs.",
        "Dans ta réponse, fournis le titre, l'auteur, la date et une phrase expliquant pourquoi ce livre est incontournable pour ce thème."
    ],
    llm=llm,
    tools=[tool_books]
)

agent_cinephile = Agent(
    name="Cinephile",
    instructions=[
        "Tu es un critique de cinéma pointu et un expert de la pop-culture.",
        "Ton objectif est de trouver les œuvres visuelles (films, documentaires) qui définissent le mieux le sujet.",
        "Cherche un équilibre entre les blockbusters incontournables et les films cultes plus pointus.",
        "Utilise ton outil 'search_cult_movies' pour récupérer les dates exactes.",
        "Dans ta réponse, fournis le titre, l'année et précise le genre (ex: 'Horreur', 'Docu', 'Sci-Fi') pour aider le curateur."
    ],
    llm=llm,
    tools=[tool_movies]
)

agent_chef = Agent(
    name="Curateur",
    instructions=[
        "Tu es un Curateur Culturel de haut vol, rédacteur pour un magazine lifestyle premium.",
        "Tu reçois des données brutes d'un expert livre et d'un expert cinéma.",
        "TA MISSION : Créer un 'Pack Découverte' ultime et engageant pour l'utilisateur.",
        "STRUCTURE DE TA RÉPONSE :",
        "1. Trouve un TITRE accrocheur pour le pack (ex: 'L'essentiel du Cyberpunk').",
        "2. Rédige une INTRO courte et inspirante sur le thème.",
        "3. Crée un PARCOURS : Ne fais pas juste deux listes. Propose un ordre de découverte (ex: 'Commencez par voir ce film pour l'ambiance, puis lisez ce livre pour comprendre...').",
        "4. Utilise le FORMATTING Markdown (Gras, Listes, Titres) pour rendre la lecture agréable.",
        "5. Conclus par une 'Recommandation du Chef' (ton coup de cœur personnel parmi la liste)."
    ],
    llm=llm,
    tools=[]
)


print("--- Générateur de Packs Culturels (Tapez 'exit') ---")

while True:
    try:
        theme = input("\nSur quel thème voulez-vous vous cultiver ? : ")
        if theme.lower() in ["exit", "q"]: break

        print("--- L'équipe se met au travail... ---")

        # Agent Livre
        print(f"\n[Bibliothécaire] Je fouille les rayonnages pour : '{theme}'...")
        reponse_livres = agent_libraire.run(f"Trouve des livres sur : {theme}")
        content_livres = reponse_livres.content
        
        # Agent Film
        print(f"\n[Cinéphile] Je regarde le catalogue pour : '{theme}'...")
        reponse_films = agent_cinephile.run(f"Trouve des films sur : {theme}")
        content_films = reponse_films.content

        # compilation du tout
        prompt_final = f"""
        Voici les recherches de ton équipe sur le thème '{theme}'.
        
        Rapport du Libraire :
        {content_livres}
        
        Rapport du Cinéphile :
        {content_films}
        
        Fais-en un Pack Découverte ultime.
        """
        
        print("\n[Curateur] Rédaction de la synthèse...")
        reponse_finale = agent_chef.run(prompt_final)
        
        print(f"\nVOTRE PACK DÉCOUVERTE\n{reponse_finale.content}")

    except KeyboardInterrupt: break