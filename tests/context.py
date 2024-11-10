# -*- coding: utf-8 -*-

import sys
import os
import dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
dotenv.load_dotenv(override=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
