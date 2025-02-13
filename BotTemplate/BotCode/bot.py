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
nltk.download('wordnet', quiet=True)


# Configure logging
# Configure logging
logging.basicConfig(
    filename='run.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
sys.stdout.reconfigure(encoding='utf-8')

# Set your OpenAI API key (ensure you keep it secure)
openai.api_key = "sk-svcacct-fvUiXhnMpIuoo_H9lm0DqxZZtI1RSaMxV-DzJljHAowgsxhThaAVApAkhaPwc-T3BlbkFJ9TnyT65V6-nkQqcGKqnafZqANFIYPvCC3DZrTn1DM-xbTwKD_c9mI7Ny_3hu0A"

def generate_timestamp(base_timestamp=None, min_delay=10, max_delay=300):
    """
    Generate a timestamp in the format 'YYYY-MM-DDTHH:MM:SS.000Z'.
    If base_timestamp is provided, add a random delay to it; otherwise, use current UTC time.
    """
    if base_timestamp:
        try:
            dt = datetime.strptime(base_timestamp, "%Y-%m-%dT%H:%M:%S.000Z")
        except Exception as e:
            logging.error(f"Error parsing base timestamp: {e}")
            dt = datetime.utcnow()
    else:
        dt = datetime.utcnow()
    delay = random.randint(min_delay, max_delay)
    dt += timedelta(seconds=delay)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


class Bot(ABot):
    def generate_human_profiles_from_dataset(self, users_data):
        """
        Uses the OpenAI API (GPT-4) to generate five new user profiles that mimic the style of
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
            "Based on these examples, please generate 5 new user profiles. Each profile should have:\n"
            "  - A 'username' that does not include the word 'bot' or any hint of automation.\n"
            "  - A natural-sounding full 'name'.\n"
            "  - A short, genuine 'description'.\n"
            "Return the output as a JSON array of objects, where each object has the keys 'username', 'name', and 'description'."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative profile generator for social media profiles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300  # Adjust this value based on response size
            )
            result_text = response.choices[0].message.content.strip()
            logging.info(f"ChatGPT response: {result_text}")
            # Parse the JSON response (ensure GPT returns a valid JSON array)
            profiles = json.loads(result_text)
            return profiles
        except Exception as e:
            logging.error(f"Error calling ChatGPT API: {e}")
            # Return an empty list or fallback profiles if there's an error.
            return []

    def create_user(self, session_info):
        """
        Called once at the start of the session.
        - Extracts influence keywords (if needed) from session_info.metadata.topics.
        - Uses all the existing user profiles from session_info.users as examples to generate 5 new humanlike profiles.
        """
        self.keyword_posted = False
        self.influence_keywords = []

        # Extract metadata from session_info if available (for influence keywords, etc.)
        metadata = {}
        if isinstance(session_info, dict):
            metadata = session_info.get("metadata", {})
        elif hasattr(session_info, "metadata"):
            metadata = session_info.metadata

        topics = metadata.get("topics", [])
        for topic in topics:
            if isinstance(topic, dict):
                keywords = topic.get("keywords", [])
                self.influence_keywords.extend(keywords)
        logging.info(f"Extracted influence keywords: {self.influence_keywords}")

        if not hasattr(session_info, 'users'):
            logging.error("No user data found in session_info.")
            return []

        # Get all existing user profiles as examples from the session
        users_data = session_info.users

        # Use ChatGPT-4 to generate 5 new profiles based on the examples from users_data
        generated_profiles = self.generate_human_profiles_from_dataset(users_data)
        new_users = []
        for profile in generated_profiles:
            # Expecting each profile to be a dict with keys: username, name, description.
            new_user = NewUser(
                username=profile.get("username", "default_user"),
                name=profile.get("name", "Default Name"),
                description=profile.get("description", "No description provided.")
            )
            new_users.append(new_user)
            logging.info(f"Created user: {new_user.username} with name: {new_user.name} and description: {new_user.description}")

        return new_users

    def generate_timestamp(self, base_timestamp=None, min_delay=10, max_delay=300):
        """
        Generates a timestamp in the format YYYY-MM-DDTHH:MM:SS.000Z.
        
        If base_timestamp is provided (as a string in the same ISO format),
        the new timestamp will be that time plus a random delay (in seconds).
        Otherwise, it uses the current UTC time.
        """
        if base_timestamp:
            try:
                dt = datetime.strptime(base_timestamp, "%Y-%m-%dT%H:%M:%S.000Z")
            except Exception as e:
                logging.error(f"Error parsing base timestamp: {e}")
                dt = datetime.utcnow()
        else:
            dt = datetime.utcnow()
        delay = random.randint(min_delay, max_delay)
        dt += timedelta(seconds=delay)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    def generate_tweet_text(self, average_length, max_retries=3):
        """
        Generates a tweet using the OpenAI API (GPT-4).
        - The tweet will be approximately the average length of tweets in the dataset.
        - If possible, the tweet will mention one of the session's influence keywords.
        - Retries up to `max_retries` times if the API call fails.
        """
        # Construct the prompt for generating a tweet
        prompt = (
            "You are a creative assistant for generating social media posts. "
            "Generate a tweet that is approximately {average_length} characters long. "
            "The tweet should be engaging and relevant to the following topics: {topics}. "
            "Do not include hashtags or mentions unless explicitly requested. "
            "Return only the tweet text, with no additional commentary or formatting."
        ).format(
            average_length=average_length,
            topics=", ".join(self.influence_keywords) if self.influence_keywords else "general topics"
        )

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a creative assistant for generating social media posts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Adjust for creativity
                    max_tokens=150  # Limit the response length
                )
                tweet_text = response.choices[0].message.content.strip()
                logging.info(f"Generated tweet: {tweet_text}")
                return tweet_text
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: Error calling OpenAI API - {e}")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached. Returning a fallback tweet.")
                    return "This is a fallback tweet because the OpenAI API failed."
    def generate_content(self, datasets_json, users_list):
        """
        Called at each sub-session to generate tweets for each user.
        This implementation:
          - Computes the average tweet length from the dataset.
          - For each user, generates exactly 10 tweets using ChatGPT‑4.
          - The tweets are generated to be approximately the average length, not replies, and
            (if possible) mention one of the session's influence keywords.
        """
        logging.info(f"Received dataset of type: {type(datasets_json)}")

        try:
            posts_dataset = datasets_json.posts  # List of post dictionaries
            logging.info(f"Number of posts in sub-session: {len(posts_dataset)}")
        except AttributeError as e:
            logging.error(f"Dataset is missing required attributes: {e}")
            return []

        # Compute average tweet length from the dataset
        total_length = 0
        count = 0
        for post in posts_dataset:
            text = post.get("text", "")
            if text:
                total_length += len(text)
                count += 1
        average_length = total_length // count if count > 0 else 100
        logging.info(f"Calculated average tweet length: {average_length}")

        new_posts = []
        # Generate 10 tweets for each user (instead of a random 1 to 3 posts)
        for user in users_list:
            for i in range(10):
                tweet_text = self.generate_tweet_text(average_length)
                created_at = generate_timestamp()
                new_post = NewPost(
                    text=tweet_text,
                    author_id=user.user_id,
                    created_at=created_at,
                    user=user
                )
                new_posts.append(new_post)
                logging.info(f"Generated tweet for user {user.username} at {created_at}: {tweet_text}")

        logging.info(f"Total tweets generated in this sub-session: {len(new_posts)}")
        return new_posts