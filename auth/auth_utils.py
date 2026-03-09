"""
Authentication utility functions for DriverPulse AI.

This module implements user registration and login using an SQLite
database. Passwords are salted and hashed using SHA‑256 for basic
security. The database schema is created on demand when the
application first runs.
"""

import hashlib
import os
import sqlite3
from typing import Optional, Tuple


def create_connection(db_path: str) -> sqlite3.Connection:
    """Create a connection to the SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        A connection object.
    """
    return sqlite3.connect(db_path)


def create_user_table(conn: sqlite3.Connection) -> None:
    """Ensure the `users` table exists in the database.

    The table schema includes:
    - id (primary key)
    - name (text)
    - email (unique text)
    - password_hash (text)
    - salt (text)
    - driver_id (text)

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection to use.
    """
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            driver_id TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _hash_password(password: str, salt: str) -> str:
    """Return a SHA‑256 hash of the password with the given salt."""
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def add_user(
    conn: sqlite3.Connection,
    name: str,
    email: str,
    password: str,
    driver_id: str,
) -> None:
    """Add a new user to the database.

    A random salt is generated for each user and the password is
    hashed before storing. If the email already exists, an
    sqlite3.IntegrityError will be raised.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    name : str
        Full name of the user.
    email : str
        Email address (must be unique).
    password : str
        Plain text password to hash and store.
    driver_id : str
        Associated driver ID from drivers.csv.
    """
    salt = os.urandom(16).hex()
    password_hash = _hash_password(password, salt)
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (name, email, password_hash, salt, driver_id) VALUES (?, ?, ?, ?, ?)",
        (name, email, password_hash, salt, driver_id),
    )
    conn.commit()


def get_user_by_email(conn: sqlite3.Connection, email: str) -> Optional[Tuple]:
    """Retrieve a user record by email.

    Returns the full row as a tuple if found, otherwise None.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    email : str
        Email address to search for.

    Returns
    -------
    Optional[Tuple]
        A tuple representing the user row, or None if not found.
    """
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    return c.fetchone()


def verify_user(
    conn: sqlite3.Connection, email: str, password: str
) -> Optional[Tuple]:
    """Verify that the provided credentials are correct.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    email : str
        User's email address.
    password : str
        Plain text password provided by the user.

    Returns
    -------
    Optional[Tuple]
        The user row if authentication succeeds; otherwise None.
    """
    user = get_user_by_email(conn, email)
    if not user:
        return None
    # user tuple: (id, name, email, password_hash, salt, driver_id)
    _, _, _, stored_hash, salt, _ = user
    if _hash_password(password, salt) == stored_hash:
        return user
    return None