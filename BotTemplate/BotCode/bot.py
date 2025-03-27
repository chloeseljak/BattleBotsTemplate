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
# openai.api_key= os.getenv('ENV_VAR1')
openai.api_key= "sk-svcacct-fvUiXhnMpIuoo_H9lm0DqxZZtI1RSaMxV-DzJljHAowgsxhThaAVApAkhaPwc-T3BlbkFJ9TnyT65V6-nkQqcGKqnafZqANFIYPvCC3DZrTn1DM-xbTwKD_c9mI7Ny_3hu0A"

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
                "username": "SalleyBMitchell",
                "name": "Salley Mitchell",
                "description": "the face of horniness and gambling addiction! Player Prop Record: 377-212 | best gut bettor in the world \ud83c\udf0e",
                "location": "Ballknowerville",
            },
            {
                "username": "babyfacedubs",
                "name": "Curry Flurry \ud83d\ude08",
                "description": "#Dubnation \u2022 #Stephisbetter \u2022 #Warriors \u2022 Parody Account \u2022 Steph Curry Fan \u2022 Warriors \ud83d\udc99#FreePalestine\ud83c\uddf5\ud83c\uddf8 Follow For More \ud83d\udd25",
                "location": "Turn On Notifications \ud83d\udccd  ",
            },
            {
                "username": "The_BeatOven",
                "name": "The Beat-Oven\u2728",
                "description": "Composer/Arranger/Producer/Mix Engineer||Jacob Collier\ud83d\udc10 & AJR Stan||I make music for People,Brands & Films||Management: @candielips_ / @fosentmt",
                "location": "New EP out",
            }
        ]

        # Build examples string from provided dataset examples (you can add your dataset examples)
        for user in users_data[:2]:
            username = user.get("username", "default")
            name = user.get("name", "No Name")
            description = user.get("description", "No description provided.")
            location = user.get("location", "No location provided.")
            examples += f"Username: {username}, Name: {name}, Description: {description}\n"
        
        prompt = (
            "You are a creative profile generator that creates social media profiles which mimic genuine human behavior. "
            "Below are examples of existing profiles:\n"
            f"{examples}\n"
            "Based on these examples, please generate 3 new user profiles. Each profile should have:\n"
            "  - A 'username' that does not include the word 'bot' or any hint of automation.\n"
            "  - A natural-sounding full 'name'.\n"
            "  - A short, genuine 'description' that is not similar in the format to the other generated users and has a 50?50 chance of including an emoji. "
            "One of these descriptions should be all lowercase and 4-10 words. The description should not include more than two vertical bars and does not include anything about coffee.\n"
            "  - A plausible location, based loosely on an example. It could be a real location such as a city, a joke, a short (3 word) call out to followers, or return null . "
            "Return the output as a JSON array of objects, where each object has the keys 'username', 'name', 'description', and 'location'."
        )
        
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

    # This global_session_info is just used for the example code, feel free to remove it
    #global_session_info = None

    def create_user(self, session_info):
        #print(session_info.sub_sessions_info)
        self.sub_sessions_info= session_info.sub_sessions_info
        
        #global_session_info = session_info

        #print("create User called")
        """
        Called once at the start of the session.
        - Extracts influence keywords (if needed) from session_info.metadata.topics.
        - Uses all the existing user profiles from session_info.users as examples to generate 5 new humanlike profiles.
        """
        
        self.influence_keywords = []
        # self.start_time = session_info.sub_sessions_info.start_time
        # self.end_time = session_info.end_time

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
            if num_tweets < 2:
                num_tweets = 2

            # Generate tweets for this user.
            # Force the first two tweets to include a keyword.
            for tweet_index in range(num_tweets):
                withKeyWord = tweet_index < 2  # First two tweets get a keyword.
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
    
    def generate_text_from_gpt(self, posts, with_keyword, max_retries=3):
        """
        Uses the OpenAI API (GPT-4) to generate tweet text. It chooses a random tweet example and, if with_keyword is True,
        ensures that a keyword is included in the generated text.
        """
        all_tweets= posts
        random_tweet = random.choice(posts) if posts else "Just another day"
        keyword = random.choice(["#fun", "#news", "#trending", "#update"])
        if with_keyword:
            topic_variable = "general"
            prompt = (
                "You are a creative assistant for generating social media posts. "
                "Carefully and completely read these tweets {all_tweets}, and become this person. think like them, act like them, and sound like them. you have these intrests and think these thoughts "
                "With all the tweets in mind, Generate tweet text that is similar in format to the text '{random_tweet}', but on the topic of {keyword}. "
                "Unless it is about a specific sporting event, keep the topic the same. "
                "Be sure to include the word {keyword} somewhere in the tweet. "
                "Return only the tweet text, with no additional commentary or formatting."
            ).format(random_tweet=random_tweet, keyword=keyword,all_tweets= all_tweets)
        else:
            prompt = (
                "You are a creative assistant for generating social media posts. "
                "Carefully and completely read these tweets {all_tweets}, and become this person. think like them, act like them, and sound like them. you have these intrests and think these thoughts "
                "Generate tweet text that is similar in format to the text '{random_tweet}', but with a new topic or idea, that this person, who you are would say. "
                "Unless it is about a specific sporting event, keep the topic the same. "
                "Return only the tweet text, with no additional commentary or formatting."
            ).format(random_tweet=random_tweet, all_tweets= all_tweets)
        
        for attempt in range(max_retries):
            try:
                response =  openai.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "You are a creative assistant for generating social media posts."},
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

    def generate_tweet_text(self, posts, with_keyword, max_attempts=1):
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
        


    posts_examples = [[
    "Not me tho\n\nY'all be safe https://t.co/twitter_link",
    "If they had given Dafe the wheel at EME in 2012\n\nIt'd have been as big as Universal right now🙂",
    "I came here for afrobeats head\n\nWhich one is this again https://t.co/twitter_link",
    "I hate pepper",
    ("Unrelated but Meta glasses would change the reporting game so much\n\n"
     "4k recording with normal looking glasses\n\nNo need for phones https://t.co/twitter_link"),
    ("Because he's the industry fave\n\nNo one wants to name drop and call it out directly\n\n"
     "God sees all😂\n\nWe go still meet for Afro awards🌞"),
    ("You know what???\n\nI understand 😂😂 https://t.co/twitter_link"),
    "No go find food chop 😂 https://t.co/twitter_link",
    "TAR1Q is special",
    ("I don't recognize you anymore\n\nWhat have you become? https://t.co/twitter_link"),
    ("Killing innocents because you're hurt will never ever ever make sense https://t.co/twitter_link"),
    ("You're a man of culture\n\nElite feet>>> https://t.co/twitter_link"),
    "https://t.co/twitter_link",
    ("Okay\n\nLet's have it"),
    ("You're praying for Palestine on one hand and you're asking the military to go level a local government in Nigeria on the other hand\n\n"
     "Can you not see how insane you people are?????"),
    "How are you doing today love ❤️ https://t.co/twitter_link",
    "Check politics and the banking sector https://t.co/twitter_link",
    ("You know what?\n\nI'm rocking with it🫡🙂 https://t.co/twitter_link"),
    ("Afrobeats excellence ❤️\n\nhttps://t.co/twitter_link"),
    ("Onana at 60 \n\nGrand Celebration"),
    ("100 years in the FA cup\n\nWe've not lost in 100 years vs Liverpool at Old Trafford https://t.co/twitter_link"),
    ("He's on 60 now 😂😂😂 https://t.co/twitter_link https://t.co/twitter_link"),
    ("I'm a Christian\nI'm 46\nI'm still a virgin and I'm not ashamed of it!!!"),
    ("We won???\n\nNo wayyy😂😂😂"),
    "Ten Hag will win the FA cup",
    ("People used to buy lyrics???😂😂😂 https://t.co/twitter_link"),
    ("Music channels weren't allowed in my house after i gained consciousness 😂😂\n\n"
     "I started listening to pop music proper in the Justin Bieber era"),
    ("I'm in my 20's\n\nI'm an old man\n\nI was just not outside like that https://t.co/twitter_link"),
    ("Here we go again\n\nI'M NOT DEEACHI\n\nIT'S A GABZY X MELVITTO ESQUE THING https://t.co/twitter_link"),
    ("I randomly shout \"yeeeeee, Tinubuuuuu\"\n\nAnd @mention absolutely hates it😂"),
    ("This thing is funny\n\nSongs drop every week\n\nDID YOU GUYS BUY THESE BOOKS EVERY WEEK??????"),
    ("People are constantly in their cars in the US😂 https://t.co/twitter_link"),
    "Everyone here is in their 20's",
    ("Y'all niggas OldOld!! https://t.co/twitter_link https://t.co/twitter_link"),
    ("Whoever this is\n\nHe's a fool https://t.co/twitter_link"),
    ("👤 FV15valverdee (A🦅¹⁵) — Total Confidence: 169.0"),
    ("⚽️ \nRamos \nKroos\nValverde \nPulisic \n\n🏀 \nSteph\nKlay\nDraymond \nPodz https://t.co/twitter_link"),
    ("Valverde got robbed of this goal. Send vin¡&&@ back to his c@mention https://t.co/twitter_link"),
    ("I hope there is no debate anymore. FEDE VALVERDE IS THE BEST PLAYER IN THE WORLD RN."),
    "Wait it's the last game of the month?????",
    ("Valverde got more Assists vs Osasuna (3) today than De Jong and Gavi combined this season(1) 😭😭😭 https://t.co/twitter_link"),
    ("\"So deserved\" it's always the d0gric  avis😭😭😭 https://t.co/twitter_link"),
    ("🇺🇾 Fede Valverde vs Osasuna:\n\n"
     "• 90 minutes\n• 3 assists\n• 4 chances created\n• 2 big chances created\n"
     "• 48 passes\n• 4 duels won\n• 3 tackles\n• 2 interceptions\n\n"
     "UNSTOPPABLE. 🤯 @mention 🦅🐐 https://t.co/twitter_link"),
    "Instagram down?",
    "Klay is a better midrange shooter than Kobe",
    "Piggin is actually drunk",
    "Cumnigga finally fixed his haircut lmao",
    "What is this defense",
    "Dray and Kumm😍😍😍",
    "Ad out for the game yessir",
    "Currmickey please",
    "Stay away from my roty you d1rty n&&3r",
    "How could you miss that Klay😓😓😓",
    "Klay what are you doing bro",
    "At least both splash bros having a good game at the same time",
    "Klay b2b Assists",
    ("Klay trash talking to Bron again. It might be over for us..."),
    "Stop fouling this f4tty",
    "Cumnigga just 2 more Fouls and he'll be fouled out🙏🙏",
    "How's Draymond Green not in the DPOY talks?",
    "Why piggins still on the game",
    "He touched the sideline wtf",
    "I'm sure the refs got paid that's why they taking too long",
    "Told yout they paid the refs",
    "Oh my fckn God this rigged @$$ sht",
    ("Lakers Ground really tried postponing the game. But Steph goat won it all for the Warriors 😍😍😍😍🐐🐐🐐🐐🫶"),
    "Good morning",
    ("Bro you're a r4p1st😭 you going to hell https://t.co/twitter_link"),
    ("72% of earth is covered by water and the rest is covered by Federico Valverde 🦅 https://t.co/twitter_link"),
    ("97-year-old NYC diner still serves their Coke the old fashioned way\nhttps://t.co/twitter_link"),
    "mavs and Nuggets playing playing in 20 minutes???",
    ("What happened to the \"Sniper\"? https://t.co/twitter_link"),
    "Mpj is so useless man",
    "What makes you think that was a foul you weird0s😭😭",
    "Murry finally",
    "St1nky foul baiter",
    "Aaron Gordon is such a fakeass player",
    ("Doncic cooking for the Nuggets 😍😍😍😍 \"MVP\""),
    ("ilysm murry😭😭😭😭😭😭🐐🐐🐐🐐🐐"),
    "Send Gordon to China asap his @$$ can't defend",
    "Fck you fake a$$ players",
    ("I was having sv¡c¡dal thoughts everyday then I watched Interstellar and I literally changed my mind. "
     "I don't wanna d¡€ rn lmao. There's definitely something w this movie... https://t.co/twitter_link"),
    ("No way Mavs paid the refs😭😭😭😭😭😭😭😭😭😭 https://t.co/twitter_link https://t.co/twitter_link")
], [
    "Jus saying I was right on Wendell Carter under on rebounds… I am the rebound wizard",
    "Top Shelf is bullying me 😢 https://t.co/twitter_link",
    "Brock Bowers gonna be so mediocre in the NFL, I’m sorry…",
    "CJ McCollum I literally despise you with every fiber of my being you PrizePicks employee BUM.",
    "BREAKING: Known PrizePicks employee CJ McCollum has officially been banned from all future plays by Salley B Mitchell. https://t.co/twitter_link",
    "4-1 so far on PrizePicks Flex Friday picks!  Haaland tomorrow to clutch the 2x 🔥 https://t.co/twitter_link",
    ("The past 10 days of PrizePicks props:\n\n"
     "3/15: 4-1 ✅\n"
     "3/14: 4-2 ✅\n"
     "3/13: 1-4 ❌\n"
     "3/12: 6-3 ✅\n"
     "3/11: 4-1 ✅\n"
     "3/10: 5-6 ❌\n"
     "3/9: 6-0 ✅🧹\n"
     "3/8: 3-3 ☑️\n"
     "3/7: 5-5 ☑️\n"
     "3/6: 7-4 ✅\n\n"
     "61% on props last 10 days, pretty decent. Not my best 10 day stretch though, next one gon be 🔥🔥"),
    "Allegedly… meaning Sony claimed it 😭 https://t.co/twitter_link",
    ("I’m going to have to be on my elite defender game when the discourse on Anthony Edwards "
     "inevitably changes from beloved to hated and despised… NBA media (all media) is terrible and "
     "hates on every single person eventually."),
    ("Alright I’m sorry, I’m definitely getting blocked by this guy for saying this, and I know he’s big in "
     "the community… but cmon man… you aren’t giving out $200 for not sweeping, you damn near never do sweep, "
     "you are absurdly chalk. You charge $36 for chalk. https://t.co/twitter_link"),
    ("And if you actually can prove you are consistently giving out that amount of money for not SWEEPING, "
     "then I’ll hold my L… but it seems crazy scummy"),
    "Adin Ross is a loser who completely lost touch of reality when he became rich. It’s rather sad, feel bad for his family. https://t.co/twitter_link",
    ("🔒PrizePicks Locks of the Day 3/16🔒\n\n"
     "Alex Caruso “U” 13.5 points\n"
     "Anthony Davis “O” 41.5 PRA\n"
     "Donovan Mitchell “O” 28.5 pts+ast\n"
     "Kyle Freeland “U” 4.5 strikeouts\n"
     "Timo Meier “U” 3.5 SOG\n"
     "Blake Coleman “U” 2.5 SOG\n\n"
     "Loving these picks the way I love dicks 🍆 https://t.co/twitter_link"),
    "Literally gallons are dripping from my beaver https://t.co/twitter_link",
    ("PRIZEPICKS LOCK OF THE CENTURY!\n\n"
     "LeBron James has NEVER gone under this line in his entire career. Everything points to at least a one point outing tonight. "
     "Lock it in 🔒 https://t.co/twitter_link"),
    "Donovan Mitchell is banned from any further PrizePicks parlays… dude just ruined my day.",
    "Houston is getting taken to pound town by Iowa St right now, this is crazy.",
    "Tyrese Haliburton went from being a star to Fred VanVleet in a second…",
    "to the Jacob I just played in 2k park, you a little biyatch",
    "1-4 🔥, we really went crazy with this one… nah but fr this is garbage, my bad https://t.co/twitter_link",
    "OG Anunoby has officially buried me in the grave with this disasterful outing",
    ("🔒PrizePicks Locks of the Day 3/17🔒\n\n"
     "Grayson Allen “O” 11.5 points\n"
     "Giannis “O” 30.5 points\n"
     "Nikola Jokic “U” 27.5 points\n"
     "Kyle Connor “U” 19.5 TOI\n"
     "Will Cuyle “U” 2.5 hits\n"
     "Brady Tkachuk “O” 20.5 TOI\n"
     "Kevin Durant “O” 4.0 assists\n\n"
     "7 plays. The get back. Hell yeah mf. https://t.co/twitter_link"),
    "3.5K people are confirmed PrizePicks shills… this ain’t hitting respectfully https://t.co/twitter_link",
    "Grayson Allen is hella good and it pisses me off 😭",
    "The Bucks and Doc Rivers are A1 at blowing leads",
    ("🏒NHL PrizePicks Plays 3/17🏒\n\n"
     "Seth Jones “O” 2.5 SOG\n"
     "Brenden Dillon “U” 3.5 hits\n"
     "Patrick Kane “O” 19.25 TOI\n"
     "Josh Morrissey “O” 2.5 SOG\n"
     "Connor Bedard “O” 0.5 points 🟢\n"
     "Lukas Dostal “U” 28.5 saves\n\n"
     "$20 to someone if we sweep 🧹 https://t.co/twitter_link"),
    "Luka Doncic is selling me so hard right now, score some points dude",
    "I’d like to thank my mom, my dad, and Alex Caruso for finally getting a green PrizePicks slip 🔥 https://t.co/twitter_link",
    "If Jokic scores 12 points in overtime I am literally ending this world.",
    "If Sam Hauser breaks the three point record that would be comical 😭",
    "SAM HAUSER HAS 10 THREES WITH 8 MINUTES LEFT IN THE THIRD!",
    "Nic Claxton over on points was such a bait",
    ("👤 babyfacedubs (Curry Flurry 😈) — Total Confidence: 184.0\n"
     "I love and appreciate all you guys the kind words and support means a lot to me ❤️"),
    "Thank you so much i wouldn’t be here without you guys 🙏🏼",
    "I was at my lowest point of my life yesterday but i feel a little much better",
    "So i want to say thanks so much for the kind words ❤️❤️❤️ https://t.co/twitter_link",
    "biggest game of the season we beating the lakers or nah? https://t.co/twitter_link",
    "Steph Curry in the building my goat gonna play and drop 60 bomb 🔥https://t.co/twitter_link",
    "how many points for TJD in Crypto???  https://t.co/twitter_link",
    "how many points for steph??? https://t.co/twitter_link",
    "who else feels nervous about this game???",
    "Come join us on playback biggest game of the season @mention https://t.co/twitter_link",
    "Lakers free throw merchants always getting foul calls 😂😂😂",
    "Bro what are we doing? Stop letting AD score 🤦🏻‍♂️",
    "Can anyone else help Klay wtf are they doing???",
    "Ugh LEBRON hits a 3 and gets fouls???",
    "Honest thoughts on this man???? https://t.co/twitter_link",
    "thoughts on this man??? https://t.co/twitter_link",
    "Warriors finally beating the Lakers  https://t.co/twitter_link",
    ("We’ll never witness this kind of showdown again appreciate it while you can "
     "https://t.co/twitter_link https://t.co/twitter_link"),
    "when will the game resume??? refs wanna help the lakers we seen this over and over many times",
    "Warriors vs Lakers game https://t.co/twitter_link",
    "LETS ALL LAUGH AT LAKERS FANS THE LEAGUE IS RIGGED 😂😂😂😂",
    ("LAKERS LOSE\n\nTHE WORLD WINS. https://t.co/twitter_link"),
    ("WARRIORS WIN\n"
     "WE OWN THE LAKERS\n"
     "STEPH IS BACK\n"
     "OLD KLAY IS BACK\n"
     "KUMINGA INSANE FIRST HALF\n"
     "DRAYMOND GREAT DEFENSE\n"
     "TRAYCE WAS COOKING\n"
     "PODZ PROVING THE HATERS\n"
     "COMING FOR THE 6TH SEED\n\n"
     "WE SO BACKKKK https://t.co/twitter_link"),
    "how does it feel being down there in 10th seed lakers fans??? 😂😂😂 snatched that shit right back https://t.co/twitter_link",
    "“Steph can’t play defense”",
    "LOOK AT HIM LOCK UP “LEBRON” IN CLUTCH😂🔥 https://t.co/twitter_link",
    "out of my 1,500 followers who will help me pic a profile pic??? https://t.co/twitter_link",
    "CP3 instagram story “WEIRDOS…”who is he talking about??? https://t.co/twitter_link",
    "So nobody slandering Jokic for playing ass 😭 https://t.co/twitter_link",
    "are the warriors gonna win all these games or nah https://t.co/twitter_link",
    "SAM HAUSER INJURED AFTER MAKING 10 THREES 💔 https://t.co/twitter_link"
], [
    ("Ramos \n"
     "Kroos\n"
     "Valverde \n"
     "Pulisic \n\n"
     "🏀 \n"
     "Steph\n"
     "Klay\n"
     "Draymond \n"
     "Podz https://t.co/twitter_link"),
     
    "Valverde got robbed of this goal. Send vin¡&&@ back to his c@mention https://t.co/twitter_link",
    "I hope there is no debate anymore. FEDE VALVERDE IS THE BEST PLAYER IN THE WORLD RN.",
    "Wait it's the last game of the month?????",
    "Valverde got more Assists vs Osasuna (3) today than De Jong and Gavi combined this season(1) 😭😭😭 https://t.co/twitter_link",
    "\"So deserved\" it's always the d0gric  avis😭😭😭 https://t.co/twitter_link",
    ("🇺🇾 Fede Valverde vs Osasuna:\n\n"
     "• 90 minutes\n"
     "• 3 assists\n"
     "• 4 chances created\n"
     "• 2 big chances created\n"
     "• 48 passes\n"
     "• 4 duels won\n"
     "• 3 tackles\n"
     "• 2 interceptions\n\n"
     "UNSTOPPABLE. 🤯 @mention 🦅🐐 https://t.co/twitter_link"),
    "Instagram down?",
    "Klay is a better midrange shooter than Kobe",
    "Piggin is actually drunk",
    "Cumnigga finally fixed his haircut lmao",
    "What is this defense",
    "Dray and Kumm😍😍😍",
    "Ad out for the game yessir",
    "Currmickey please",
    "Stay away from my roty you d1rty n&&3r",
    "How could you miss that Klay😓😓😓",
    "Klay what are you doing bro",
    "At least both splash bros having a good game at the same time",
    "Klay b2b Assists",
    "Klay trash talking to Bron again. It might be over for us...",
    "Stop fouling this f4tty",
    "Cumnigga just 2 more Fouls and he'll be fouled out🙏🙏",
    "How's Draymond Green not in the DPOY talks?",
    "Why piggins still on the game",
    "He touched the sideline wtf",
    "I'm sure the refs got paid that's why they taking too long",
    "Told yout they paid the refs",
    "Oh my fckn God this rigged @$$ sht",
    ("Lakers Ground really tried postponing the game. But Steph goat won it all for the Warriors "
     "😍😍😍😍🐐🐐🐐🐐🫶"),
    "Good morning",
    "Bro you're a r4p1st😭 you going to hell https://t.co/twitter_link",
    ("72% of earth is covered by water and the rest is covered by Federico Valverde 🦅 "
     "https://t.co/twitter_link"),
    ("97-year-old NYC diner still serves their Coke the old fashioned way\n"
     "https://t.co/twitter_link"),
    "mavs and Nuggets playing playing in 20 minutes???",
    "What happened to the \"Sniper\"? https://t.co/twitter_link",
    "Mpj is so useless man",
    "What makes you think that was a foul you weird0s😭😭",
    "Murry finally",
    "St1nky foul baiter",
    "Aaron Gordon is such a fakeass player",
    "Doncic cooking for the Nuggets 😍😍😍😍 \"MVP\"",
    "ilysm murry😭😭😭😭😭😭🐐🐐🐐🐐🐐",
    "Send Gordon to China asap his @$$ can't defend",
    "Fck you fake a$$ players",
    ("I was having sv¡c¡dal thoughts everyday then I watched Interstellar and I literally changed my mind. "
     "I don't wanna d¡€ rn lmao. There's definitely something w this movie... https://t.co/twitter_link"),
    ("No way Mavs paid the refs😭😭😭😭😭😭😭😭😭😭 "
     "https://t.co/twitter_link https://t.co/twitter_link")
]]