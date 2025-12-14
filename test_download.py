import requests
import time

def test_sec_download():
    # SEC.gov requires a User-Agent header with an email
    headers = {
        "User-Agent": "SECRAGHackathon/1.0 (test@example.com)"
    }
    
    # Test with a known filing URL from the dataset (using one that looks stable)
    # Example: LIBERTY STAR GOLD CORP 8-K
    url = "https://www.sec.gov/Archives/edgar/data/1172178/000108503706002090/ex10-1.htm"
    
    print(f"Testing download from: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Success! Downloaded {len(response.text)} characters.")
            print("First 200 chars:")
            print(response.text[:200])
            return True
        else:
            print(f"Failed. Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_sec_download()
