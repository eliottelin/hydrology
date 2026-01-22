#!/usr/bin/env python3
"""
List available model IDs from the TAMUS Chat API.
Usage:
  python tamu_list_models.py --key YOUR_KEY
  python tamu_list_models.py --base-url https://chat-api.tamu.ai --key YOUR_KEY

The script does a DNS check of the host, then GETs /openai/models.
Prints a table of model ids and any available metadata.
"""

import argparse
import socket
import sys
from urllib.parse import urlparse
import requests
import json


def dns_resolve(hostname):
    try:
        ip = socket.gethostbyname(hostname)
        return ip
    except Exception as e:
        return None


def list_models(base_url, api_key):
    parsed = urlparse(base_url)
    host = parsed.hostname or base_url
    print(f"Resolving host: {host}...")
    ip = dns_resolve(host)
    if not ip:
        print(f"ERROR: Could not resolve host: {host}", file=sys.stderr)
        return 2
    print(f"Resolved: {host} -> {ip}")

    url = base_url.rstrip('/') + '/openai/models'
    headers = {
        'Authorization': f'Bearer {api_key}' if api_key else '',
        'Accept': 'application/json'
    }
    try:
        print(f"GET {url} ...")
        r = requests.get(url, headers=headers, timeout=20)
        print(f"HTTP {r.status_code}")
        try:
            j = r.json()
        except Exception:
            print('Failed to parse JSON response:')
            print(r.text[:4000])
            return 3

        # Expecting a JSON list or dict with models
        models = None
        if isinstance(j, list):
            models = j
        elif isinstance(j, dict):
            # try common keys
            if 'data' in j and isinstance(j['data'], list):
                models = j['data']
            elif 'models' in j and isinstance(j['models'], list):
                models = j['models']
            else:
                # fallback: pretty-print full dict
                print(json.dumps(j, indent=2)[:4000])
                return 0

        if not models:
            print('No models list found in response.')
            return 0

        print('\nAvailable models:')
        for m in models:
            if isinstance(m, dict):
                mid = m.get('id') or m.get('model_id') or m.get('name') or str(m)
                desc = m.get('description') or m.get('summary') or ''
                print(f"- {mid}: {desc}")
            else:
                print(f"- {m}")
        return 0
    except requests.exceptions.RequestException as e:
        print(f"Network error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='https://chat-api.tamu.ai', help='TAMU API base URL (default https://chat-api.tamu.ai)')
    p.add_argument('--key', default=None, help='TAMU API key (or set TAMU_KEY env)')
    args = p.parse_args()

    api_key = args.key
    if not api_key:
        import os
        api_key = os.getenv('TAMU_KEY')

    rc = list_models(args.base_url, api_key)
    sys.exit(rc)


if __name__ == '__main__':
    main()
