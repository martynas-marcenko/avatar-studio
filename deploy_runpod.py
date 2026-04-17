#!/usr/bin/env python3
"""Deploy Avatar Studio to RunPod Serverless."""

import os
import json
import time
import requests
from pathlib import Path

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
GITHUB_REPO = "https://github.com/martynas-marcenko/avatar-studio"
HANDLER = "runpod_handler"

if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY not set")
    exit(1)

def create_serverless_endpoint():
    """Create a RunPod serverless endpoint."""

    url = "https://api.runpod.io/graphql"

    query = """
    mutation CreateServerlessEndpoint($input: CreateEndpointInput!) {
        createServerlessEndpoint(input: $input) {
            id
            name
            status
        }
    }
    """

    variables = {
        "input": {
            "name": "avatar-studio-infinitetalk",
            "description": "InfiniteTalk avatar video generation",
            "templateId": "avatar-studio",
            "githubRepo": GITHUB_REPO,
            "githubRepoBranch": "master",
            "handler": HANDLER,
            "containerRegistryAuthId": None,
            "jobMaxRetries": 3,
            "jobTimeoutSeconds": 3600,
            "gpuCount": 1,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    payload = {
        "query": query,
        "variables": variables
    }

    print("Creating RunPod serverless endpoint...")
    print(f"Repo: {GITHUB_REPO}")
    print(f"Handler: {HANDLER}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "errors" in data:
            print(f"Error: {data['errors']}")
            return None

        endpoint = data.get("data", {}).get("createServerlessEndpoint", {})

        if endpoint:
            print(f"✓ Endpoint created!")
            print(f"  ID: {endpoint.get('id')}")
            print(f"  Name: {endpoint.get('name')}")
            print(f"  Status: {endpoint.get('status')}")
            return endpoint
        else:
            print("No endpoint returned in response")
            print(f"Response: {json.dumps(data, indent=2)}")
            return None

    except Exception as e:
        print(f"Error creating endpoint: {e}")
        print(f"Status: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


if __name__ == "__main__":
    endpoint = create_serverless_endpoint()

    if endpoint:
        print("\n✓ Deployment initiated!")
        print(f"\nNext steps:")
        print(f"1. Go to https://www.runpod.io/console/serverless to monitor deployment")
        print(f"2. Once deployed, you'll get an API endpoint URL")
        print(f"3. Update Avatar Studio to use the remote endpoint")
    else:
        print("\n✗ Deployment failed")
        exit(1)
