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
nltk.download('wordnet', quiet=True)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stdout.reconfigure(encoding='utf-8')

class Bot(ABot):
    def create_user(self, session_info):
        self.keyword_posted = False
        self.influence_keywords = []
        
        metadata = {}
        if isinstance(session_info, dict):
            metadata = session_info.get("metadata", {})
        elif hasattr(session_info, "metadata"):
            metadata = session_info.metadata

        topics = metadata.get("topics", [])
        # Loop over each topic and extract its keywords.
        for topic in topics:
            if isinstance(topic, dict):
                keywords = topic.get("keywords", [])
                self.influence_keywords.extend(keywords)
        """
        This function creates a list of bot users.
        It modifies usernames using the modify_username method.
        """
        if not hasattr(session_info, 'users'):
            logging.error("No user data found in session_info.")
            return []

        new_users = []
        
        # Process up to 5 users for variety
        for user_data in session_info.users[:5]:  
            original_username = user_data.get("username", "default")
            
            # Generate a modified username using the class method
            new_username = self.modify_username(original_username)
            
            new_user = NewUser(
                username=new_username,
                name=user_data.get("name", "Default Name"),
                description=user_data.get("description", "No description provided.")
            )
            new_users.append(new_user)
            logging.info(f"Created user: {new_user.username}")

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

    def generate_content(self, datasets_json, users_list):
        """
        Called at each sub-session to generate posts.
        
        This implementation selects a random post from the sub-session's posts.
        If the selected post's text begins with '@', it is assumed to be a reply,
        and the new post's timestamp is set to a random delay after the original post's timestamp.
        Otherwise, the timestamp is generated based on the current time.
        """
        logging.info(f"Received dataset of type: {type(datasets_json)}")

        try:
            posts_dataset = datasets_json.posts  # List of post dictionaries
            users_dataset = datasets_json.users  # Not used in this example, but available if needed
            logging.info(f"Number of posts in sub-session: {len(posts_dataset)}")
            logging.info(f"Number of users in sub-session: {len(users_dataset)}")
        except AttributeError as e:
            logging.error(f"Dataset is missing required attributes: {e}")
            return []

        # Create a copy of the posts dataset so we can remove posts as we use them.
        available_posts = posts_dataset.copy()

        new_posts = []
        for user in users_list[:1]:
            # Decide randomly how many posts to generate in this sub-session (e.g., 1 to 3 posts)
            num_posts = random.randint(1, 3)
            for _ in range(num_posts):
                if available_posts:
                    # Pick a random original post from the available posts
                    original_post = random.choice(available_posts)
                    # Remove the post from the list to avoid reuse
                    available_posts.remove(original_post)
                    text = original_post.get("text", "Default text")
                    # If this is a reply (text starts with '@'), use the original post's time as base.
                    if text.strip().startswith("@"):
                        base_time = original_post.get("created_at")
                        created_at = self.generate_timestamp(base_timestamp=base_time)
                    else:
                        created_at = self.generate_timestamp()
                else:
                    text = "Default fallback post if no text available."
                    created_at = self.generate_timestamp()

                new_post = NewPost(
                    text=text,
                    author_id=user.user_id,
                    created_at=created_at,
                    user=user
                )
                new_posts.append(new_post)
                for kw in self.influence_keywords:
                    if kw.lower() in text.lower():
                        self.keyword_posted = True
                        break

                logging.info(f"Generated post for user {user.username} at {created_at}: {text}")
                # If after generating posts no keyword has been used, force one post with a keyword
        
        if not self.keyword_posted and self.influence_keywords:
            chosen_keyword = random.choice(self.influence_keywords)
            forced_text = f"Let's talk about {chosen_keyword}!"
            forced_post = NewPost(
                text=forced_text,
                author_id=users_list[0].user_id,
                created_at=self.generate_timestamp(),
                user=users_list[0]
            )
            new_posts.append(forced_post)
            self.keyword_posted = True
            logging.info(f"Forced keyword post: {forced_text}")

        logging.info(f"Total new posts generated in this sub-session: {len(new_posts)}")
        return new_posts
    

    def get_synonym(self, word):
        """
        Uses WordNet to get a synonym for a word.
        If no synonym is found (or it returns the same word), the original word is returned.
        """
        synsets = wordnet.synsets(word)
        if synsets:
            # Collect synonyms from the first synset
            synonyms = [lemma.name() for lemma in synsets[0].lemmas() if lemma.name().lower() != word.lower()]
            if synonyms:
                return random.choice(synonyms)
        return word

    def modify_username(self, original_username):
        """
        Breaks the original username into parts (splitting on underscores, dots, or spaces),
        attempts to replace longer parts with a synonym, and then reassembles the username
        with a random separator. Optionally, adds a random number at the end.
        """
        # Split on underscores, dots, or whitespace
        parts = re.split(r'[_\.\s]', original_username)
        new_parts = []
        
        for part in parts:
            if not part:
                continue
            
            # If part contains digits, leave them as is, otherwise consider synonym replacement
            if part.isalpha() and len(part) > 3:
                # Get a synonym for the word; if not found, original is returned.
                new_word = self.get_synonym(part)
            else:
                new_word = part
            
            # Capitalize the new part (for display style)
            new_parts.append(new_word.capitalize())
        
        # Randomly choose a separator: underscore, dot, or no separator (for a different look)
        separator = random.choice(["_", ".", ""])
        new_username = separator.join(new_parts)
        
        # Optionally, add a random two-digit number at the end to mimic structures like T2_Free2
        if random.random() < 0.5:
            new_username += str(random.randint(10, 99))
        
        return new_username

  