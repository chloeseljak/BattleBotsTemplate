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
                "username": "319E17th",
                "name": "Boss Clown",
                "description": "Sticking to sports is treason.",
                "location": "Old Ohio",
            },
            {
                "username": "LunaMuse",
                "name": "Luna Harper",
                "description": "always dreaming in color",
                "location": "Stargazer's Point",
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
            "You are an imaginative profile generator tasked with creating authentic social media profiles that sound genuinely human. "
            "Carefully read the 3 human examples here {examples}"
            "Create 3 new profiles, basing each profile directly on these examples \n\n"
            "1. Each profile must include:\n"
            "   - A 'username' that sounds natural and does not contain the word 'bot' or any reference to automation.\n"
            "   - A realistic full 'name' (first and last name).\n"
            "   - A brief 'description' that feels personal and genuine. In 50% of the cases, include one emoji naturally. "
            "     One of these profiles must have its description entirely in lowercase and contain between 4 and 10 words. "
            "     Also, ensure that no description has more than two vertical bars ('|') and avoid any mention of coffee.\n"
            "   - A 'location' that could be a real city, a country, a playful or humorous location, a short (up to three words) shout-out to followers, or null.\n\n"
            "2. Make sure the profiles vary in style and format so that they do not look too similar to each other.\n\n"
            "Return the result as a JSON array of objects, where each object has the keys: 'username', 'name', 'description', and 'location'."
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
                "You are a creative social media assistant with a knack for capturing a specific person's style. "
                "Below are several tweets from this individual: {all_tweets} "
                "Study these tweets carefully to understand their tone, word choices, interests, and personality. "
                "Now, generate a new tweet that is similar in style and format to the following example: '{random_tweet}'. "
                "Your tweet must address the topic of {keyword}—and it must include the word '{keyword}' somewhere in the text. "
                "If the topic isn't related to a specific sporting event, keep the overall subject consistent. "
                "Return only the tweet text, with no additional commentary or formatting."
            ).format(random_tweet=random_tweet, keyword=keyword,all_tweets= all_tweets)
        else:
            prompt = (
                "You are a creative social media assistant with an exceptional ability to mimic a specific person's writing style. "
                "Below are a series of tweets written by this individual: {all_tweets} "
                "Study these tweets carefully to understand every nuance of their voice—pay attention to their tone, word choices, sentence structure, humor, and overall personality. "
                "Notice the topics they discuss, how they express emotions, and the cultural or contextual references they include. "
                "Your task is to generate a new tweet that is indistinguishable from one that this person would write. "
                "Make sure the tweet captures the same energetic or reflective tone, fits with their established style, and aligns with their interests and thoughts as reflected in the examples. "
                "The tweet should be engaging, authentic, and consistent with the voice of the provided examples. "
                "Return only the tweet text with no additional commentary, formatting, or explanation."
            ).format(all_tweets= all_tweets)
        
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
        
    posts_examples = [['   Fred VanVleet   has officially  sent me  into  oblivion  with  this  catastrophic performance.', 'Why is nobody boxing out on rebounds? Come on team, hustle up! 🤦🏻\u200d♂️', " The past 10 days of NBA takes:\n\n3/15: Steph MVP case ✅\n3/14: Lakers overhyped ❌\n3/13: Warriors defense on point ✅\n3/12: Jokic not clutch ❌\n3/11: LeBron still the king ✅\n3/10: AD can’t carry ❌\n3/9: Klay back to form ✅\n3/8: Nets contenders? ☑️\n3/7: Bucks unstoppable ☑️\n3/6: Grizzlies underrated ✅\n\nSolid 60% hit rate on takes, but I'm ready to step it up next time! 🔥🏀", 'Jordan  Poole  under  on assists was such a trap.', 'Jalen Green went from being the next big thing to just another player in the blink of an eye…', ' PRIZEPICKS DOMINATION  \n  \nWE  HIT   BIG TODAY  \n  \n  \nCARUSO   CAME THROUGH  \n  \nANTHONY  DAVIS ON FIRE  \n  \n  \nMITCHELL   REDEEMED  HIMSELF  \n  \nFREELAND  UNDER LOCKED IN  \n  \nCOLEMAN  WITH  THE  CLUTCH  \n\nWE  SO  UP RIGHT  NOW 🔥🔥  https://t.co/twitter_link', ' BREAKING: Known gamer Adin Ross has officially been banned from all future online lobbies by Salley B Mitchell. https://t.co/twitter_link\n', 'LOOK AT HIM BREAKING ANKLES ON THE COURT, UNSTOPPABLE 😂🔥 https://t.co/twitter_link', "Julius   Randle   I   literally   can't  stand your  shot   selection,   it's   like   you're  allergic  to  passing  the  ball.", "If  you can actually prove LeBron isn't benefiting from all these phantom calls, then I'll admit I'm wrong… but it seems pretty sketchy. ", "Ben Simmons I literally cannot stand how you play every game like it's practice, you overpaid bench warmer.", 'PRIZEPICKS  DOMINATION   \nMY PICKS WERE ON FIRE  \nCARUSO CASHED IN  \nANTHONY DAVIS DOMINATED  \nMEIER WAS A LOCK  \nCOLEMAN CAME THROUGH  \nJOKIC PROVED THE DOUBTERS WRONG  \n\nFEELING UNSTOPPABLE 🔥 https://t.co/twitter_link', "“Giannis can't shoot threes”", '🏀NBA PrizePicks Plays 3/18🏀\n\nJayson Tatum “O” 27.5 points\nBam Adebayo “U” 9.5 rebounds\nKyrie Irving “O” 6.5 assists\nZion Williamson “U” 1.5 steals\nKlay Thompson “O” 4.5 threes\nChris Paul “U” 18.5 points+assists\n\n$20 to someone if we sweep 🧹 https://t.co/twitter_link', 'Ugh JOKIC misses another layup???', 'Literally drowning in my own thoughts right now 😂 https://t.co/twitter_link ', 'The last 10 days of NBA drama:\n\n3/15: Lakers lose 😂\n3/14: Warriors win 🔥\n3/13: Klay shines 🌟\n3/12: LeBron struggles ❌\n3/11: Steph goes off ✅\n3/10: AD dominates 🤦🏻\u200d♂️\n3/9: Warriors sweep 🧹\n3/8: Lakers choke 🥴\n3/7: Curry clutch ✅\n3/6: Lakers fall short ❌\n\nWarriors are on fire, next week’s gonna be even crazier! 🔥🏀 ', 'We’ll never see a player like Steph again, savor every splash while you can 🎯🏀 https://t.co/twitter_link https://t.co/twitter_link', "LeBron's latest post is just him smirking… what's he hinting at??? 😂🔥 https://t.co/twitter_link", 'Do we really trust Harden in the playoffs???????????? 🤔 https://t.co/twitter_link', " 4K people are officially part of the Fantasy Football hype… this season's gonna be wild respectfully 😤 https://t.co/twitter_link", 'DAMIAN    LILLARD DROPS  50  WITH  6  MINUTES   LEFT  IN THE  FOURTH!', '🏀NBA PrizePicks Plays 3/18🏀 Jayson Tatum “O” 27.5 points Devin Booker “U” 6.5 assists Klay Thompson “O” 3.5 threes\nBam Adebayo “U” 10.5 rebounds Trae Young “O” 8.5 assists Julius Randle “U” 25.5 points $50 to someone if we sweep 🧹 https://t.co/twitter_link', '3-0 so far on the weekend picks! Messi tomorrow to seal the hat-trick 🔥 https://t.co/twitter_link ', 'LETS ALL LAUGH AT THE KNICKS FANS THINKING THEY HAD A CHANCE 😂😂😂😂', ' Miami is getting absolutely demolished by Boston right now, this is wild.'], ['Sounds like a whackin’s afoot! https://t.co/twitter_link', 'Sorry guys. @mention \n\nhttps://t.co/twitter_link.', 'Everyone knows that the Gays hate boobs. https://t.co/twitter_link', 'Beauty for Fields in Pittsburgh is they have no investment in the “starter.”  No reason he can’t compete and win that job.', 'Just found out my daughters are graduating on Star Wars Day 2025. Are you available to speak, @mention @mention?', 'AHL product, @mention prices. Go Jackets! \n#CBJ', 'Curious, do you track xga by goal or know someone who does, @mention? i.e. the average xg on actual goals for Elvis Merzlikins.', 'Shocked that a lineup including Brendan Gaunce, Mathieu Olivier, Jake Christianson, Trey Fix-Wolansky, Carson Meyer, and Michael Pyythia  looks underwhelming against an NHL lineup. #CBJ', 'OTOH, I want to leave; OTOH I want to boo this team at the end. #CBJ https://t.co/twitter_link', 'I bring the fam down twice or thrice a year just to keep hope alive, cos I’m the only one who cares. \n\nI’m out on #CBJ hockey. I can’t justify why we’re even here.', 'We suck. 5 at least of the forwards tonight are AHL talents. Elvis has allowed 5 goals on maybe 2.5 XG? https://t.co/twitter_link'], ['I wish I could remember who wrote it, I read an article last year sometime talking about how America is an incredibly convoluted series of overlapping systems - and once even one of these industries begins to spiral, it will unravel the entire thing in catastrophic fashion https://t.co/twitter_link', 'The shipping industry is responsible for import and export of goods. Logistics industry transports these goods all over the country. Healthcare, education, economic industries all rely on logistics to get the required physical materials to function.', 'Shipping and logistics industries rely on Healthcare and economic industries to keep workers healthy and educated etc etc', 'If you wanna shift the goalpost a little to make him look even worse you could go back to 2013 when he praised China\'s "basic dictatorship"\n\nhttps://t.co/twitter_link https://t.co/twitter_link', "Lil Sparkle Socks looks like he's on the verge of tears https://t.co/twitter_link https://t.co/twitter_link", "Planned Parenthood and other abortion providers are the modern day Temple of Moloch and people like you are it's priests and priestesses https://t.co/twitter_link", "Wasn't there a security guard on 9/11 who had a bad feeling about one of the literal hijackers but also didn't want to appear racist and waved him through? https://t.co/twitter_link", 'It was a big deal at my school when we got a computer lab with a couple of these bad boys in it https://t.co/twitter_link https://t.co/twitter_link', 'Lil Sparkle Socks when he has to face the consequences of his actions \n\n(Leaving an event out of the back door because of a pro-Palestine mob in the front) https://t.co/twitter_link https://t.co/twitter_link', "Lil Sparkle Socks looks like he's on the verse of tears leaving an event out of the back door because of a pro-Palestine mob out front\n\nThink he's starting to realize the consequences of his actions yet? https://t.co/twitter_link https://t.co/twitter_link", 'Trevor from Black Dahlia. Brother lived and breathed metal and genuinely loved people and fans. \n\nHe also showed love to Christian metal bands when most people shit on them; I might be in the minority but that meant a lot\n\nHe was a fuckin good dude who was taken from us too soon https://t.co/twitter_link', 'This is cheating but I also wanna say David Gold. He spoke to so many people and yet Woods of Ypres is still so obscure; virtually everyone who finds his work resonates with it, the man communicated on a different level from all of us, that we all can relate to https://t.co/twitter_link', "Keep getting your boosters, friend. It's in the best interests of the public https://t.co/twitter_link https://t.co/twitter_link", "Did I misunderstand the question?\n\nI don't think it's gotten any harder for Trudeau's children to function.\n\nCanada thought they done with Pierre, yet here we are. Who's to say 40 years from now Canadians aren't cursing the next Trudeau all over again? https://t.co/twitter_link", "Everyone talking like you're coming back from this. There is no coming back from $34T in debt and $300B being added daily. \n\nThey are intentionally crashing the economy to force everyone into the new digital one built on the block chain.\n\nhttps://t.co/twitter_link https://t.co/twitter_link", "100%\n\nOnly cure is to turn off (I write from my device I've been browsing for the last 15 minutes)\n\nSeriously though, the further we can get away from our devices, the better. I'm seriously considering switching back to a flip phone https://t.co/twitter_link"]]