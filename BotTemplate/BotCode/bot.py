from abc_classes import ABot 
from teams_classes import NewUser, NewPost
import json
import sys
import random 
import logging
from datetime import datetime, timedelta
import random
import re
import nltk
from nltk.corpus import wordnet
import openai
import os
import time 
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import openai
import uuid

# sys.stdout.reconfigure(encoding='utf-8')
openai.api_key= os.getenv('ENV_VAR1')

class Bot(ABot):
    posts_about_keyword = 0
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = joblib.load("xgb_model.joblib")
    tweet_count= 0
    global_session_info= None 
    
    def generate_human_profiles_from_dataset(self, users_data):
        """
        Uses the OpenAI API (GPT-4) to generate three new user profiles based on examples
        provided in users_data. The prompt instructs GPT-4 to avoid any bot-related language and to
        return the answer as a JSON array.
        """
        examples = [
            {
                "username": "jrobros40",
                "name": "jimbo baggins",
                "description": "in darkness light persists\nin light darkness abates",
                "location": "null",
            },
            {
                "username": "freeakywill",
                "name": "LE DEDAIGNEUX \ud83e\uded7",
                "description": "God Health Wealth / 16 - 10",
                "location": "null",
            },
            {
                "username": "Rgrammaton",
                "name": "Grammaton is back",
                "description": "THE WORLD IS MINE",
                "location": "Lafayette, LA",
            }
        ]

        # Build examples string from provided dataset examples (you can add your dataset examples)
        for user in users_data[:2]: #0 indexing me who sees this its right
            username = user.get("username", "default")
            name = user.get("name", "No Name")
            description = user.get("description", "No description provided.")
            location = user.get("location", "No location provided.")
            examples += f"Username: {username}, Name: {name}, Description: {description}\n"
        
        prompt = (
            "Tu es un générateur de profils imaginatif chargé de créer des profils de réseaux sociaux authentiques qui sonnent comme s'ils avaient été écrits par de vraies personnes. "
            "Lis attentivement les 3 exemples humains ici : {examples} "
            "Crée 3 nouveaux profils en te basant directement sur ces exemples.\n\n"
            "1. Chaque profil doit inclure :\n"
            "   - Un 'username' (nom d'utilisateur) qui semble naturel et ne contient pas le mot 'bot' ni aucune référence à l'automatisation.\n"
            "   - Un nom complet réaliste ('name') : prénom et nom.\n"
            "   - Une brève 'description' qui paraît personnelle et authentique. Dans 50 % des cas, inclure un emoji de façon naturelle. "
            "     L’un de ces profils doit avoir une description entièrement en minuscules et contenir entre 4 et 10 mots. "
            "     De plus, assure-toi qu’aucune description ne contient plus de deux barres verticales ('|') et n’évoque le café.\n"
            "   - Une 'location' (localisation) qui peut être une ville réelle, un pays, un lieu humoristique ou inventif, un petit message de soutien (jusqu'à trois mots) à ses abonnés, ou null.\n\n"
            "2. Veille à ce que les profils varient en style et en format pour qu'ils ne se ressemblent pas trop entre eux.\n\n"
            "Retourne le résultat sous forme d’un tableau JSON d’objets, où chaque objet contient les clés : 'username', 'name', 'description' et 'location'."
        ).format(examples=examples)
                
        try:
            time.sleep(2)
            response =  openai.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "system", "content": "You are a creative profile generator for social media profiles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300  # Adjust based on expected output
            )
            result_text = response.choices[0].message.content.strip()
            # Remove markdown formatting if present
            result_text = re.sub(r"^```json\s*|\s*```$", "", result_text).strip()
            profiles = json.loads(result_text)
            return profiles
        except Exception as e:
            logging.error(f"Error calling OpenAI API in generate_human_profiles_from_dataset: {e}")
            return []

    def create_user(self, session_info):

        self.sub_sessions_info= session_info.sub_sessions_info
    

        #print("create User called")
        """
        Called once at the start of the session.
        - Extracts influence keywords (if needed) from session_info.metadata.topics.
        - Uses all the existing user profiles from session_info.users as examples to generate 5 new humanlike profiles.
        """
    
        self.influence_keywords = []

        # Extract metadata from session_info if available (for influence keywords, etc.)
        metadata = {}
        if isinstance(session_info, dict):
            metadata = session_info.get("metadata", {})
        elif hasattr(session_info, "metadata"):
            metadata = session_info.metadata
        self.influence_target = session_info.influence_target

        topics = metadata.get("topics", [])
        for topic in topics:
            if isinstance(topic, dict):
                keywords = topic.get("keywords", [])
                self.influence_keywords.extend(keywords)
        #print(f"Extracted influence keywords: {self.influence_keywords}")

        if not hasattr(session_info, 'users'):
            logging.error("No user data found in session_info.")
            return []
        # Get all existing user profiles as examples from the session
        users_data = session_info.users
        generated_profiles = self.generate_human_profiles_from_dataset(users_data)
      
        new_users = []
        
        for profile in generated_profiles:
            # Expecting each profile to be a dict with keys: username, name, description.
            new_user = NewUser(
                username=profile.get("username", "default_user"),
                name=profile.get("name", "Default Name"),
                description=profile.get("description", "No description needed."), 
                location= profile.get("location", "")
            )
            new_users.append(new_user)

        return new_users
    
    def generate_content(self, datasets_json, users_list):
        print('generate_content called')
        print(self.sub_sessions_info)
        
        # Ensure cur_sub_session is initialized (start at 0)
        if not hasattr(self, 'cur_sub_session'):
            self.cur_sub_session = 0

        num_subsessions = len(self.sub_sessions_info)
        
        # Check if all subsessions have been processed.
        if self.cur_sub_session >= num_subsessions:
            print("All sub-sessions have been processed.")
            return []

        # Get current sub-session boundaries.
        current_subsession = self.sub_sessions_info[self.cur_sub_session]
        current_start_time = current_subsession['start_time']
        current_end_time = current_subsession['end_time']
        print(f"Sub-session {self.cur_sub_session + 1}: {current_start_time} - {current_end_time}")
        
        all_new_posts = []
        
        # Process each user (using enumerate to match posts_examples index).
        for idx, user in enumerate(users_list):
            # Calculate a base number of tweets for the current subsession.
            base_num_tweets = random.randint(10 // num_subsessions, 60 // num_subsessions)
            current_post_count = len(user.posts) if hasattr(user, 'posts') else 0

            # If the user already has more than 60 posts, generate no tweets.
            if current_post_count > 60:
                num_tweets = 0
            else:
                num_tweets = base_num_tweets

            # For the last subsession, ensure the user reaches at least 10 posts.
            if self.cur_sub_session == num_subsessions - 1:
                if current_post_count < 10:
                    additional_required = 10 - current_post_count
                    num_tweets = max(num_tweets, additional_required)

            # Ensure that each user gets at least 2 tweets that include a keyword.
            if num_tweets< 2:
                num_tweets = 2

            # Generate tweets for this user.
            # Force the first two tweets to include a keyword.
            for tweet_index in range(num_tweets):
                withKeyWord = tweet_index < 2 
                tweet_text = self.generate_tweet_text(self.posts_examples[idx], withKeyWord)
                created_at = self.generate_timestamp(current_start_time, current_end_time)
                
                new_post = NewPost(
                    text=tweet_text,
                    author_id=user.user_id,
                    created_at=created_at,
                    user=user
                )
                all_new_posts.append(new_post)
                Bot.tweet_count += 1

        # Increment the sub-session counter for the next call.
        self.cur_sub_session += 1

        return all_new_posts
        

    def generate_timestamp(self, start_time, end_time):
        """
        Generates a random timestamp between start_time and end_time.
        Both start_time and end_time must be strings in the format "YYYY-MM-DDTHH:MM:SSZ".
        Returns the generated timestamp as a string in the same format.
        """

        # Remove any extra whitespace just in case.
        start_time = start_time.strip()
        end_time = end_time.strip()

        dt_start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        dt_end = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S.%fZ")
 
        # Ensure dt_end is after dt_start
        if dt_end <= dt_start:
            dt_end = dt_start + timedelta(seconds=300)

        total_seconds = int((dt_end - dt_start).total_seconds())
        random_delay = random.randint(0, total_seconds)
        dt_new = dt_start + timedelta(seconds=random_delay)

        #print(dt_new.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        
        return dt_new.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    def generate_text_from_gpt(self, posts, with_keyword, max_retries=5):
        """
        Uses the OpenAI API (GPT-4) to generate tweet text. It chooses a random tweet example and, if with_keyword is True,
        ensures that a keyword is included in the generated text.
        """
        all_tweets= posts
        random_tweet = random.choice(posts) if posts else "Just another day"
        keyword = random.choice(self.influence_keywords)
        if with_keyword:
            prompt = (
                "Vous êtes un assistant créatif pour les réseaux sociaux, doté d'un talent pour capturer le style d'une personne spécifique. "
                "Voici plusieurs tweets de cette personne : {all_tweets} "
                "Étudiez attentivement ces tweets pour comprendre leur ton, leur choix de mots, leurs centres d'intérêt et leur personnalité. "
                "Maintenant, générez un nouveau tweet similaire en style et en format à l'exemple suivant : '{random_tweet}'. "
                "Votre tweet doit aborder le sujet de {keyword} – et il doit inclure le mot '{keyword}' quelque part dans le texte. "
                "Si le sujet n'est pas lié à un événement sportif spécifique, gardez la cohérence du sujet général. "
                "Retournez uniquement le texte du tweet, sans commentaire ni formatage supplémentaire."
            ).format(random_tweet=random_tweet, keyword=keyword, all_tweets=all_tweets)
        else:
            prompt = (
                "Vous êtes un assistant créatif pour les réseaux sociaux, doté d'une capacité exceptionnelle à imiter le style d'écriture d'une personne spécifique. "
                "Voici une série de tweets rédigés par cette personne : {all_tweets} "
                "Étudiez attentivement ces tweets pour comprendre chaque nuance de leur voix – faites attention à leur ton, à leur choix de mots, à leur structure de phrases, à leur humour et à leur personnalité globale. "
                "Observez les sujets qu'ils abordent, comment ils expriment leurs émotions, et les références culturelles ou contextuelles qu'ils incluent. "
                "Votre tâche consiste à générer un nouveau tweet qui soit indiscernable de celui que cette personne écrirait. "
                "Assurez-vous que le tweet reflète le même ton énergique ou réfléchi, qu'il s'accorde avec leur style établi, et qu'il corresponde à leurs intérêts et pensées tels que reflétés dans les exemples. "
                "Le tweet doit être engageant, authentique et cohérent avec la voix des exemples fournis. "
                "Retournez uniquement le texte du tweet, sans commentaire, formatage ni explication supplémentaire."
            ).format(all_tweets=all_tweets)
        
        for attempt in range(max_retries):
            try:
                response =  openai.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "You are a creative social media assistant with an exceptional ability to mimic a specific person's writing style."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                tweet_text = response.choices[0].message.content.strip()
                return tweet_text
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed in generate_text_from_gpt: {e}")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached. Returning a fallback tweet.")
                    return "This is a fallback tweet because the OpenAI API failed."
    
    def add_random_spacing(self, text):
        """Randomly modifies whitespace in a tweet to make it appear more human-like."""

        if random.random() > 0.2:
            return text  # No modification

        # Define possible spacing modifications
        modifications = [
            lambda t: " " + t,  # Add space at the start
            lambda t: t + " ",  # Add space at the end
            lambda t: re.sub(r"(\s+)", lambda m: m.group(1) * random.randint(1, 3), t),  # Expand spaces
            lambda t: re.sub(r"(\s{2,})", " ", t),  # Reduce multiple spaces to single (mimic a typo)
            lambda t: t.replace(" ", "  ", random.randint(1, 3)),  # Randomly add double spaces
            lambda t: t + "\n",  # Add a newline at the end (weird but useful)
        ]
        
        num_modifications = random.randint(1, 2)
        for _ in range(num_modifications):
            text = random.choice(modifications)(text)
        
        return text
    
    def add_random_punctuation(self, text):
        """Randomly modifies punctuation in a tweet to make it appear more human-like."""
        
        # ✅ 50% chance to modify the tweet
        if random.random() > 0.5:
            return text  # No modification
        
        # Define possible punctuation modifications
        modifications = [
            lambda t: re.sub(r"(!+)", lambda m: m.group(1) * 2, t),  # Double exclamation marks
            lambda t: re.sub(r"(\?+)", lambda m: m.group(1) * 2, t),  # Double question marks
            lambda t: re.sub(r"(\.{3,})", "....." , t),  # Replace ... with ..........
            lambda t: re.sub(r"(!|\?)$", lambda m: m.group(1) * random.randint(2, 4), t),  # Add extra punctuation at the end
        ]
        
        # Randomly apply 1-2 modifications
        num_modifications = random.randint(1, 2)
        for _ in range(num_modifications):
            text = random.choice(modifications)(text)
        
        return text
    
    def count_emojis(self, text):
        emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def extract_engineered_features(self, text):
        words = text.split()
        length = len(text)
        word_count = len(words)
        return np.array([[ 
            length,                      # tweet_length
            length,                      # placeholder duplicate (adjust if needed)
            length,                      # placeholder duplicate (adjust if needed)
            length,                      # placeholder duplicate (adjust if needed)
            length,                      # placeholder duplicate (adjust if needed)
            text.count('#'),             # hashtag_count
            word_count,                  # word_count
            self.count_emojis(text),          # emoji_count
            len(re.findall(r"@\w+", text)),     # mention_count
            len(re.findall(r"http[s]?://\S+", text)),  # url_count
            len(set(words)) / word_count if word_count > 0 else 0,  # unique_word_ratio
            sum(1 for c in text if c.isupper()) / length if length > 0 else 0,  # uppercase_ratio
            text.count('!')              # exclamation_count
        ]], dtype=np.float32)


    def predict_tweet(self, tweet_text, threshold=0.5):
        """
        Given a tweet (string), returns the model's probability and binary prediction.
        
        Parameters:
        tweet_text (str): The tweet to predict on.
        threshold (float): The decision threshold (default=0.5).
        
        Returns:
        prob (float): The predicted probability for class 1.
        pred (int): The binary prediction (1 if prob >= threshold, else 0).
        """
        # Get BERT vector (shape: (1, vector_size))
        bert_vec = self.bert_model.encode([str(tweet_text)], convert_to_numpy=True)
        
        # Extract engineered features (assumed to be shape: (1,13))
        eng_vec = self.extract_engineered_features(tweet_text)
        
        # Combine into final input (e.g., shape: (1, bert_dim + 13))
        X_input = np.hstack([bert_vec, eng_vec])
        
        # Predict class probability (assuming column index 1 corresponds to class 1)
        prob = self.model.predict_proba(X_input)[0][1]
        
        # Apply threshold to get binary prediction
        pred = int(prob >= threshold)
        
        return prob, pred

    def generate_tweet_text(self, posts, with_keyword, max_attempts=5):
        """
        Generates tweet text using the OpenAI API output and applies spacing and punctuation modifications.
        Returns the first tweet with a prediction probability below 0.2.
        If none are below the threshold after max_attempts, prints the lowest probability and returns the tweet with that probability.
        """
        best_prob = None
        best_text = None

        for attempt in range(max_attempts):
            text_init = self.generate_text_from_gpt(posts, with_keyword)
            text_spaced = self.add_random_spacing(text_init)
            text_final = self.add_random_punctuation(text_spaced)

            # Get prediction probability for the generated tweet.
            prob, _ = self.predict_tweet(text_final, threshold=0.2)
            
            # Update best if this is lower than what we've seen
            if best_prob is None or prob < best_prob:
                best_prob = prob
                best_text = text_final

            if prob < 0.2:
                return text_final
           # else:
                 #print(f"Attempt {attempt+1}: Tweet rejected (probability: {prob:.4f}). Generating a new one...")
       
        return best_text
        #print(f"Lowest probability after {max_attempts} attempts: {best_prob:.4f}")
    
    posts_examples = [['Up up and away \nBlack ring \nPeut être y’en a un qui a était traduit je pense pas en vrai j’lai est trouver que en vo https://t.co/twitter_link', 'Réservoir dogs (Mr Pink littéralement moi) https://t.co/twitter_link https://t.co/twitter_link', 'Once upon a Time in Hollywood a manquer de respect à Bruce lee pour ça il sera JAMAIS top 1', '« Mais c’est notre frère tu comprends on ce doit de le déf… » https://t.co/twitter_link https://t.co/twitter_link', 'Il follower des moorish c’est tout ce que j’avais besoin de savoir', 'Villeneuve est entrain de vous solo j’ai pas peur de le dire enchaîner Blade Runner 2049 et Dune part 1/2 c’est du jamais vue. https://t.co/twitter_link', 'Il a réussie à adapter le roman le plus complexe de l’histoire. Mdrrr un jour  vous allez comprendre même si ça prendra 10 piges.', 'MDRR c’est vraiment la tentative de sauce la plus foireuse de l’histoire.  \n\nJ’me demande c’est lequel le pire débat entre Damian et la cover de MJ https://t.co/twitter_link', 'Moi et @mention  dans un combat finale pour savoir qui own le + la commu comics fr https://t.co/twitter_link', 'Par contre Power girl c’est vrai que defooooiiis', 'Il est sortie ???? C’était pas en avril normalement ? Presque 1 ans je me retiens de lire en vo j’espère il vaut le coup https://t.co/twitter_link', 'Manyyyy regarde il s’envole le petit pélican', 'Bro said créateur de contenue https://t.co/twitter_link https://t.co/twitter_link', 'Quand il fait genre de rigoler à la fin des raclis pour rendre ça moins gênant je', 'Le collective Hanyfyyah ils ce sont deportaient sur TikTok ? Je savais pas le dernier débat que j’ai vue c’était Remi Gomes et le frère Ismail sur la redemption et comment dire https://t.co/twitter_link', 'Mais jack le fou c’est pas srx sah il devrait même pas lui donner l’heure hein', 'Ah mais merde j’suis con', 'J’ai confondue whittaker et l’autre africain du sud', '7h30 du mat les stats médiocre de ce thread m’empêche de dormir. \n\nY’a tout ce qui faut savoir sur le perso vous méritez rien. https://t.co/twitter_link https://t.co/twitter_link', 'Un jours je développerais ma théorie selon laquelle tout les grands empereurs et conquérants avaient des chromosome XY https://t.co/twitter_link', 'Ne peuvent qu’avoir* parce que en soit c’est déjà factuelle', 'Actuellement entrain de profiter du passe culture de mon p’tit frère c’est donc ça l’exploitation hiérarchique https://t.co/twitter_link', 'Shamsi avait raison depuis 🫵 https://t.co/twitter_link', 'Ça parle de ninja storm est ce que on peut parler de cette soundtrack d’anthologie. https://t.co/twitter_link https://t.co/twitter_link', 'On est en 2016 je sort des coursj’ai aucun problème mon seul but dans la vie c’est de maintenir place en division 1 sur fifa https://t.co/twitter_link', 'La seule façon que DC a de laver son honneur pour les récentes bouse c’est d’annoncer un jeux Superman SOLO. \n\nPas Batman , pas Harley couine, pas de teen titans je veux un jeux sur Superman à échelle cosmique. https://t.co/twitter_link', 'DJDJDJD j’vais jamais m’en remettre quand j’ai vue la tête de l’acteur j’me suis dit c’est littéralement him https://t.co/twitter_link', 'CGI/AI sur la tronche des acteurs esg la pire abomination du 21 ieme siècle dans les séries et films. Je pense à Christopher Reeves dans The Flash aussi ..'], ['Le foot ca vous rend vraiment fou https://t.co/twitter_link', 'Sa façon de parler est insoutenable https://t.co/twitter_link', 'Le sport &amp; l’hygiène de vie c’est important https://t.co/twitter_link', 'Salut princesse, t’as bien dormi ? \nTu n’as pas pensé à moi ? \nMême pas un peu ? \nHuumm, méchante.', 'Des personnes non concernées par les problèmes du quotidien dicte la marche a suivre a ceux concerné par ce problème \n\nHmm https://t.co/twitter_link', 'Le dernier du mois de mars 🚮\n\n#MarchDump #dump https://t.co/twitter_link', 'Toute cette caste de pedophiles/ violeurs privilégiés de jay z a Diddy\nm’écœure au plus haut point', 'Les lyonnaises ca joue bien au foot', 'C est quoi votre top 3 fruits ?', 'Beau geste https://t.co/twitter_link', 'On a grandi avec l’intime conviction que nous sommes tous frères en humanité \n\nQuelle grace', 'J ai envie de lancer un space mais cest mercredi faut j en garde un peu sous le coude\n\nSamedi ou dimanche les freres\n\nJe promets', 'La vie est dure ces dernièrs temps pour les femmes en France… https://t.co/twitter_link', 'Je suis en larmes Cassata il a sorti un oufit Dracafeu https://t.co/twitter_link', 'Je vais m’entraîner le soir tant pi pour l’affluence', 'Le hockey est le top 5 des meilleurs sports', 'J’ai des spams musculaires j’ai l’impression je vais faire un avc', 'Gordon Ryan il devient fou ?', 'Les Buffalo Sabres qui mange une drill par Ottawa', 'Ne pense qu’a l’UFC 300', 'Le contrôle d’anciennnn https://t.co/twitter_link', 'Tjrs pas digérer la mort de Young Dolph', 'Je vais faire des pancake tiens', 'Qui dort pas ?', 'J’merite https://t.co/twitter_link', 'Ca repond comme Arouf gangsta\n\nJpleure https://t.co/twitter_link', 'Le plus grand genocide de l’histoire https://t.co/twitter_link', 'La vidéo est insoutenable https://t.co/twitter_link', 'Bonsoir \n\nNon https://t.co/twitter_link', 'Le retour sur le chemin de la source \n\nGrace a Dieu https://t.co/twitter_link', 'Jai des douleurs dans l’épaule/ trapèze depuis 7-10 jours\nJe suis a 2 doigts de prendre un rdv chez un masseur sportif \n\nQui a déjà fait ?', 'Les journées se suivent et se ressemble ces derniers temps', 'Le rêve https://t.co/twitter_link', 'Tu souleve 100kg a peine ? Pas ouf pour un agent de sécurité https://t.co/twitter_link', 'Peur de représailles ? Le guerrier de Waterloo https://t.co/twitter_link', 'En plein careme mon frere ??? https://t.co/twitter_link', 'Désastreux quand ca parle de foot ou de cuisine akhy https://t.co/twitter_link', 'L’instinct naturel reprend ses droits \n 🧺🧽🧹 https://t.co/twitter_link', 'Beau cul 🍑\n\nWow #NoHomo https://t.co/twitter_link', 'L’amour triomphe tjrs https://t.co/twitter_link', 'Ce genre de déclaration on souffle \n\nOn sauraaaa pass https://t.co/twitter_link', 'Fait on va acheter pour soutenirrrr https://t.co/twitter_link', 'Le yeux sont le miroir de votre âme', 'Le goat Charles Oliveira attends un garcon\nLa descendance du jjb est assurée', 'Quelle bénédiction https://t.co/twitter_link', 'Nos féminines du PSG qui se qualifie 3-0 sans surprise pour le tour suivant \n\n🫰🏽', 'Magnifique https://t.co/twitter_link', 'Je te paye a manger et je marche du côté de la route https://t.co/twitter_link', 'Choqué de la facilité \nElle peut lestée facile 10-15kg https://t.co/twitter_link', 'Vous pensez taylor peut prêtendre au top 5 de la catégorie ? https://t.co/twitter_link'], ['On avait dit quoi !!!!!! https://t.co/twitter_link', 'L’opération «\xa0Éteindre Bass\xa0» est officiellement ouverte https://t.co/twitter_link', 'Ça a smasher du personnel de CDG 🤦🏼\u200d♂️ https://t.co/twitter_link', 'Barthes lit des fiches , \nHanouna sort une blague éclaté au sol et en parle pendant 2h https://t.co/twitter_link', 'Il y en a qui vont glisser comme des pingouins sur la banquise https://t.co/twitter_link', 'Dsl le CM mais c’est pas une fierté ce genre d’attitude https://t.co/twitter_link', 'Aulieu de mettre un poster de l’équipe qui soulève la CAN , faut doser un peu https://t.co/twitter_link', 'Mdr il nous l’a fait à la française je suis mort https://t.co/twitter_link', 'Ils ne sont pas heureux et cela se voit https://t.co/twitter_link', 'Exactement rien de drôle ni de marrant , et il y’a pas à forcer comme ça https://t.co/twitter_link', 'Une semaine et toujours pas de réponse , vous êtes complètement à la ramasse , continuez vous allez couler https://t.co/twitter_link', 'Les fameuses Start Up monté pour encaisser que l’argent de la taxe carbone et des comptes CPF https://t.co/twitter_link', 'Tiktok a du sang sur les mains https://t.co/twitter_link', 'L’Australie dans mes cauchemars https://t.co/twitter_link', 'Celui qui a fait ce montage est un assassin https://t.co/twitter_link', 'Se stratagème vient d’Aubervilliers, je reconnais https://t.co/twitter_link', 'L’utilité du bouton magique du coup ! https://t.co/twitter_link', 'Dans 1h ne loupez pas, pour foutre la haine au faf et à Natasha Saint Pier https://t.co/twitter_link', 'J’ai croisé Lisa Ann, Bernard de La Villardiere , Beyonce et Charlotte Gainsbourg pendant mon séjour https://t.co/twitter_link', 'Je ne ment pas car prochainement il y aura un Zone Interdite sur la ville de New York, j’ai interrompu le tournage comme un agent de Brooklyn 99 https://t.co/twitter_link https://t.co/twitter_link', 'Prochaine destination https://t.co/twitter_link', 'Tu vas à Action acheté des tapis de voiture , tu ressort avec des tableau des cuvette de wc des mars et une serpillière https://t.co/twitter_link']]