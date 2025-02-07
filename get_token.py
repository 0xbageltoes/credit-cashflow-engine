from supabase import create_client
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv('NEXT_PUBLIC_SUPABASE_URL'),
    os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
)

async def sign_in():
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    
    try:
        # Sign in with email and password
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        # Get the session
        session = response.session
        if session:
            print("\nAuthentication successful!")
            print("\nYour access token (use this for testing):")
            print(session.access_token)
        else:
            print("\nNo session returned. Authentication might have failed.")
            
    except Exception as e:
        print(f"\nError during authentication: {str(e)}")

if __name__ == "__main__":
    asyncio.run(sign_in())
