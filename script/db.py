# https://sqliteviewer.app

import os
import sqlite3
import streamlit as st
import pandas as pd


db_path = os.path.abspath(os.path.join("db", "guardfall.db"))

# 1. data 폴더가 없으면 생성
if not os.path.exists("db"):
    os.makedirs("db")

# 2. SQLite 연결
def get_connection():
    conn = sqlite3.connect(db_path)
    return conn

# 3. 테이블 생성
def create_tables():
    with get_connection() as conn:
        cursor = conn.cursor()

        # Users 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT,
            medical_condition TEXT,
            contact_info TEXT
        )
        """)

        # Coordinates 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fall_detection_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            left_shoulder_x REAL,
            left_shoulder_y REAL,
            left_shoulder_v REAL,
            left_shoulder_vr REAL,
            right_shoulder_x REAL,
            right_shoulder_y REAL,
            right_shoulder_v REAL,
            right_shoulder_vr REAL,
            left_knee_x REAL,
            left_knee_y REAL,
            left_knee_v REAL,
            left_knee_vr REAL,
            right_knee_x REAL,
            right_knee_y REAL,
            right_knee_v REAL,
            right_knee_vr REAL,
            checkFall INTEGER
        )
        """)

def insert_user(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    with get_connection() as conn:
        df.to_sql('fall_detection_data', conn, if_exists='append', index=False)