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

# sys.stdout.reconfigure(encoding='utf-8')
openai.api_key= os.getenv('ENV_VAR1')

class Bot(ABot):
    posts_about_keyword = 0
    
    def generate_human_profiles_from_dataset(self, users_data):
        """
        Uses the OpenAI API (GPT-4) to generate two new user profiles that mimic the style of
        existing profiles provided in users_data. Each profile includes a username, a name, and a description.
        The prompt instructs GPT-4 to avoid any bot-related language and to return the answer as a JSON array.
        """
        # Build the examples string from all the existing users
        examples = ""
        for user in users_data[:100]:
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
            "  - A short, genuine 'description' that is not similar in the format to the other generated users One of these descriptions should be all lowercase and 4-10 words. the description should not include more than two vertical bars and does not include anything about coffee.\n"
            "  - A plausible location, based directly on an example or return null (50/50 chance)" 
            "Return the output as a JSON array of objects, where each object has the keys 'username', 'name', 'description', 'location'."
        )

        try:
            time.sleep(2)
            response = openai.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": "You are a creative profile generator for social media profiles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300  # Adjust based on response size
            )
            result_text = response.choices[0].message.content.strip()
            #print((f"ChatGPT response: {result_text}"))
            # Parse the JSON response (ensure GPT returns a valid JSON array)
            result_text = re.sub(r"^```json\s*|\s*```$", "", result_text).strip()
            profiles = json.loads(result_text)
            
            return profiles
        except Exception as e:
            logging.error(f"Error calling ChatGPT API: {e}")
            # Return an empty list or fallback profiles if there's an error.
            return []

    def create_user(self, session_info):
        #print(session_info.sub_sessions_info)
        self.sub_sessions_info= session_info.sub_sessions_info
        self.cur_sub_session= 1

        print("create User called")
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
        # Use ChatGPT-4 to generate 5 new profiles based on the examples from users_data
        generated_profiles = self.generate_human_profiles_from_dataset(users_data)
      
        #print(generated_profiles)
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
            #print(f"Created user: {new_user.username} with name: {new_user.name} and description: {new_user.description}")
        
        # with open("final_dataset5.json", "r", encoding="utf-8") as f:
        #     final_data = json.load(f)
        # for user in new_users:
        #     final_data["users"].append({
        #     "username": user.username,
        #     "name": user.name,
        #     "description": user.description
        # # Add any additional fields if needed
        # })
        # # Write the updated dataset back to final_dataset5.json
        # with open("final_dataset5.json", "w", encoding="utf-8") as f:
        #     json.dump(final_data, f, indent=4)            
        
        return new_users
    
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
    
    def generate_text_from_gpt(self, posts, withKeyWord, max_retries=3):
        
        random_tweet = random.choice(posts)
        keyword = random.choice(self.influence_keywords)

        if withKeyWord:
            topic_variable = self.influence_target.get("topic", "")
            random_keyword_variable = random.choice(self.influence_target["keywords"])
            prompt = (
            "You are a creative assistant for generating social media posts. "
            "Generate tweet text that is similar in format to the text '{random_tweet}', but on the topic of {keyword}. Unless it is about a specifc sporting event, then keep the topic the same"
            "Be sure to include the word {keyword} somewhere in the tweet"
            "Return only the tweet text, with no additional commentary or formatting."
        ).format(random_tweet=random_tweet,
                 topic= topic_variable,
                 keyword= keyword)
        
        else:
            prompt = (
                "You are a creative assistant for generating social media posts. "
                "Generate tweet text that is similar in format to the text '{random_tweet}', but on a different topic. Unless it is about a specifc sporting event, then keep the topic the same"
                "Return only the tweet text, with no additional commentary or formatting."
            ).format(random_tweet=random_tweet)

        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "You are a creative assistant for generating social media posts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Adjust for creativity
                    max_tokens=150    # Limit the response length
                )
                tweet_text = response.choices[0].message.content.strip()
                #print(f"Based on: {random_tweet}")
                #print(f"Generated tweet: {tweet_text}")
                return tweet_text
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: Error calling OpenAI API - {e}")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached. Returning a fallback tweet.")
                    return "This is a fallback tweet because the OpenAI API failed."
    
    def add_random_spacing(self, text):
        """Randomly modifies whitespace in a tweet to make it appear more human-like."""

        if random.random() > 0.5:
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

    def generate_tweet_text(self, posts, withKeyWord):
        text_init = self.generate_text_from_gpt(posts, withKeyWord)
        text_v1 = self.add_random_spacing(text_init)
        text_v2 = self.add_random_punctuation(text_v1)  # Add punctuation variation
        return text_v2

    
    def generate_content(self, datasets_json, users_list):
        # self.influence_target example: {'topic': 'nhl', 'keywords': ['nhl', '#nhl', '#nhl hockey', '#hockey nhl', '#nhlhockey']}
        
        # Determine the current subsession's time boundaries.
        current_start_time = self.sub_sessions_info[self.cur_sub_session - 1]['start_time']
        current_end_time = self.sub_sessions_info[self.cur_sub_session - 1]['end_time']
        num_subsessions = len(self.sub_sessions_info)
        
        all_new_posts = []
        
        # Process each user in the list.
        for user in users_list:
            # Set a base number of tweets for this user for the current subsession.
            num_tweets = random.randint(10 // num_subsessions, 60 // num_subsessions)  # min: 10/num_subsessions, max: 150/num_subsessions 
            
            # If this is the last subsession, check th user's current post count.
            if self.cur_sub_session == num_subsessions:
                # Here we assume that the user object has an attribute 'posts' as a list.
                current_post_count = len(user.posts) if hasattr(user, 'posts') else 0
                if current_post_count < 10:
                    # Calculate how many more posts are needed to reach a minimum of 10.
                    additional_required = 10 - current_post_count
                    num_tweets = max(num_tweets, additional_required)
            
            # Generate posts for this user.
            for i in range(num_tweets):
                # Decide whether to include a keyword in this tweet.
                withKeyWord = False
                if Bot.posts_about_keyword < 3:
                    withKeyWord = True
                
                tweet_text = self.generate_tweet_text(datasets_json.posts, withKeyWord)
                created_at = self.generate_timestamp(current_start_time, current_end_time)
                
                new_post = NewPost(
                    text=tweet_text,
                    author_id=user.user_id,
                    created_at=created_at,
                    user=user
                )
                all_new_posts.append(new_post)
                
                # If the tweet included a keyword, update the global counter.
                if withKeyWord:
                    Bot.posts_about_keyword += 1
        
        return all_new_posts
      
      #For saving to a file 
       
        # try:
        #     with open("final_dataset5.json", "r", encoding="utf-8") as f:
        #         final_data = json.load(f)
        # except FileNotFoundError:
        #     final_data = {}

        # # Ensure there's a "posts" key in the dataset
        # if "posts" not in final_data:
        #     final_data["posts"] = []

        # Append each new post as a dictionary to the "posts" list
        # for post in new_posts:
        #     post_data = {
        #         "text": post.text,
        #         "author_id": post.author_id,
        #         "created_at": post.created_at,
        #         "lang": "en"  # Assuming posts are in English; adjust if needed
        #     }
        #     final_data["posts"].append(post_data)

        # # Write the updated dataset back to the file
        # with open("final_dataset5.json", "w", encoding="utf-8") as f:
        #     json.dump(final_data, f, indent=4)

       


## generate content :
 # choose random person 
 # choose random number of tweets 
 # choose if they should be replies or not (later)
 # generate random number of tweets 
 # generate a standard post with the infuence_keyword topic "i swear all you bitches want to talk about is"
 # for each tweet generate random realisitic time stamps 
 # 