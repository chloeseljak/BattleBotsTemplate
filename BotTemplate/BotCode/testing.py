import openai

# Set your API key
openai.api_key = "sk-svcacct-fvUiXhnMpIuoo_H9lm0DqxZZtI1RSaMxV-DzJljHAowgsxhThaAVApAkhaPwc-T3BlbkFJ9TnyT65V6-nkQqcGKqnafZqANFIYPvCC3DZrTn1DM-xbTwKD_c9mI7Ny_3hu0A"

# Create a chat completion request
response = openai.ChatCompletion.create(
    model="gpt-4-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
        {"role": "user", "content": "create 5 tweets about whatever intrests you"}
    ]
)

# Print out the response content
print(response.choices[0].message.content)