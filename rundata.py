import json
from api_requests import get_session_data

def main():
    response, session_dataset = get_session_data()
    if response.status_code >= 400:
        print("Error retrieving session dataset:")
        print("Status code:", response.status_code)
        print("Response:", response.text)
    else:
        # Retrieve the raw JSON data from the response
        data = response.json()
        
        # Filter out posts with an empty id
        if "posts" in data:
            original_post_count = len(data["posts"])
            data["posts"] = [post for post in data["posts"] if post.get("id", None) != ""]
            filtered_post_count = len(data["posts"])
            print(f"Filtered out {original_post_count - filtered_post_count} posts with empty id.")
        
        # Filter out users with a z_score of 0
        if "users" in data:
            original_user_count = len(data["users"])
            data["users"] = [user for user in data["users"] if user.get("z_score", None) != 0]
            filtered_user_count = len(data["users"])
            print(f"Filtered out {original_user_count - filtered_user_count} users with z_score of 0.")
        
        # Write the filtered data to a file named "final_dataset5.json"
        with open("final_dataset5.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print("Filtered session dataset saved to final_dataset5.json.")

if __name__ == "__main__":
    main()