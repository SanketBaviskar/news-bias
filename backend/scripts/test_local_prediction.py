import requests
import time
import subprocess
import sys

# The news text provided by the user
news_text = """Trump Criticizes Somali Immigrants, Plans Increased ICE Operations
Somalia,Immigration,Fraud,Ilhan Omar,Tim Walz,Minnesota,Political Polarization

Summary from the AllSides News Team
President Donald Trump on Tuesday criticized Somali immigrants in the US, calling them "garbage", describing them as overly reliant on social programs and saying he doesn't want them in the country.

The Details: During a Cabinet meeting, Trump said Somali immigrants "contribute nothing" and described their home country as "no good." He added that Somalis should "go back to where they came from and fix it." Trump also criticized Minnesota Democratic leaders, including Rep. Ilhan Omar, who emigrated from Somalia in 1995, saying, "Ilhan Omar is garbage. She's garbage. Her friends are garbage," and Gov. Tim Walz, linking both to recently reported fraud in the state's social programs. Omar responded, "His obsession with me is creepy. I hope he gets the help he desperately needs."

For Context: The remarks follow reports that Immigration and Customs Enforcement (ICE) is planning operations in Minneapolis targeting Somali immigrants with final deportation orders. The Minneapolis-St. Paul area has the largest Somali-American population in the US, with roughly 40,000 residents born in Somalia. Recent investigations found roughly $1 billion in public funds were stolen from Minnesota's safety net programs over five years, with some fraud occurring "in pockets of Minnesota's Somali diaspora." Trump said the perpetrators "should be sent back to where they came from" and indicated he would revoke temporary protected status for roughly 700 Somali nationals. 

How the Media Covered It: Outlets on the left characterized the comments as anti-Somali, with Washington Post (Lean Left bias) calling the language "dehumanizing" and New York Times (Lean Left) describing it as "an alarming use of vulgarity from the White House against an entire community." Fox News (Right) and New York Post (Lean Right) focused on Minnesota's fraud investigations and emphasized that federal authorities say immigration enforcement targets immigration status, not race. They also included background on Omar, and Trump's claims that she's "anti-sematic." Wall Street Journal (Center) framed Trump's expanded immigration enforcement within the context of a recent shooting in DC, while Reuters (Center) highlighted Trump's broader anti-immigration rhetoric and noted that ICE operations have caused concern in immigrant communities."""

def test_prediction():
    url = 'http://127.0.0.1:5000/predict'
    try:
        response = requests.post(url, json={'news_text': news_text})
        if response.status_code == 200:
            print("Request successful!")
            # Extract prediction from JSON
            try:
                data = response.json()
                print(f"Prediction: {data.get('prediction_text')}")
                print(f"Confidence: {data.get('confidence')}")
                print(f"Explanation: {data.get('explanation')}")
            except Exception as e:
                print(f"Could not parse JSON response: {e}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    test_prediction()
