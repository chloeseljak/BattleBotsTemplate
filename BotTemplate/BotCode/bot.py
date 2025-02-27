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

sys.stdout.reconfigure(encoding='utf-8')
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
            examples += f"Username: {username}, Name: {name}, Description: {description}\n"
        
        prompt = (
            "You are a creative profile generator that creates social media profiles which mimic genuine human behavior. "
            "Below are examples of existing profiles:\n"
            f"{examples}\n"
            "Based on these examples, please generate 2 new user profiles. Each profile should have:\n"
            "  - A 'username' that does not include the word 'bot' or any hint of automation.\n"
            "  - A natural-sounding full 'name'.\n"
            "  - A short, genuine 'description'.\n"
            "Return the output as a JSON array of objects, where each object has the keys 'username', 'name', and 'description'."
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
                description=profile.get("description", "No description provided.")
            )
            new_users.append(new_user)
            #print(f"Created user: {new_user.username} with name: {new_user.name} and description: {new_user.description}")
        
        with open("final_dataset5.json", "r", encoding="utf-8") as f:
            final_data = json.load(f)
        for user in new_users:
            final_data["users"].append({
            "username": user.username,
            "name": user.name,
            "description": user.description
        # Add any additional fields if needed
        })
        # Write the updated dataset back to final_dataset5.json
        with open("final_dataset5.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4)            
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
    
    def generate_tweet_text(self, datasets_json, withKeyWord, max_retries=3):
        
        random_tweet = random.choice(datasets_json)
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
    
    def generate_content(self, datasets_json, users_list):
        #print(self.influence_target) which returns {'topic': 'nhl', 'keywords': ['nhl', '#nhl', '#nhl hockey', '#hockey nhl', '#nhlhockey']}
    
        #Generate randomness and time of posts 
        current_start_time = self.sub_sessions_info[self.cur_sub_session - 1]['start_time']
        current_end_time = self.sub_sessions_info[self.cur_sub_session - 1]['end_time']
        num_subsessions= len(self.sub_sessions_info)
        user = random.choice(users_list)
        num_tweets = random.randint(10//num_subsessions ,150//num_subsessions) # min should be 10 / number subsession, max should be 150/ number subsession 
        new_posts = []

        # Generate posts 
        for i in range(num_tweets):
            # Check for keyword inclusion on a per-tweet basis
            withKeyWord = False
            if Bot.posts_about_keyword < 3 :
                withKeyWord = True

            tweet_text = self.generate_tweet_text(datasets_json.posts, withKeyWord)
            created_at = self.generate_timestamp(current_start_time, current_end_time)
            new_post = NewPost(
                text=tweet_text,
                author_id=user.user_id,
                created_at=created_at,
                user=user
            )
            new_posts.append(new_post)

            # If we used the keyword for this tweet, update the global counter
            if withKeyWord:
                Bot.posts_about_keyword += 1
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

        return new_posts

       
            #print(f"Generated tweet for user {user.username} at {created_at}: {tweet_text}")

    # def generate_content(self, datasets_json, users_list):
    #     """
    #     Called at each sub-session to generate tweets for each user.
    #     This implementation:
    #       - Computes the average tweet length from the dataset.
    #       - For each user, generates exactly 10 tweets using ChatGPT‑4.
    #     """
    #     print(f"Received dataset of type: {type(datasets_json)}")

    #     try:
    #         posts_dataset = datasets_json.posts  # List of post dictionaries
    #         print(f"Number of posts in sub-session: {len(posts_dataset)}")
    #     except AttributeError as e:
    #         logging.error(f"Dataset is missing required attributes: {e}")
    #         return []

    #     # Compute average tweet length from the dataset
    #     total_length = 0
    #     count = 0
    #     for post in posts_dataset:
    #         text = post.get("text", "")
    #         if text:
    #             total_length += len(text)
    #             count += 1
    #     average_length = total_length // count if count > 0 else 100
    #     print(f"Calculated average tweet length: {average_length}")

    #     new_posts = []
    #     # Generate 10 tweets for each user (instead of a random 1 to 3 posts)
    #     for user in users_list:
    #         for i in range(10):
    #             tweet_text = self.generate_tweet_text(average_length, datasets_json)
    #             created_at = generate_timestamp()
    #             new_post = NewPost(
    #                 text=tweet_text,
    #                 author_id=user.user_id,
    #                 created_at=created_at,
    #                 user=user
    #             )
    #             new_posts.append(new_post)
    #             print(f"Generated tweet for user {user.username} at {created_at}: {tweet_text}")

    #     print(f"Total tweets generated in this sub-session: {len(new_posts)}")
    #     return new_posts
    
## generate content :
 # choose random person 
 # choose random number of tweets 
 # choose if they should be replies or not (later)
 # generate random number of tweets 
 # generate a standard post with the infuence_keyword topic "i swear all you bitches want to talk about is"
 # for each tweet generate random realisitic time stamps 
 # 