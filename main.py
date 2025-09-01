from dotenv import load_dotenv
load_dotenv()

import os
from tools.toolset import toolset
import tools.literature_agent

if __name__ == '__main__':
    host = os.getenv("HOST", "127.0.0.1")
    try:
        port = int(os.getenv("PORT", "8001"))
    except ValueError:
        port = 8001
    toolset.serve(host=host, port=port)