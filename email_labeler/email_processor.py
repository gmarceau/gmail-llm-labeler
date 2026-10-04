"""Email processing utilities."""

import logging
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from googleapiclient.discovery import Resource

from .gmail_utils import (
    CLASSIFICATION_METADATA_HEADERS,
    add_labels_to_email,
    fetch_emails,
    get_email_content,
    get_gmail_client,
    get_or_create_label,
    remove_from_inbox,
)


class EmailProcessor:
    """Handles email fetching and processing operations."""

    def __init__(self, gmail_client: Optional[Resource] = None, lazy_init: bool = False):
        """Initialize email processor with Gmail client.

        Args:
            gmail_client: Optional Gmail API client. If not provided, creates a new client.
            lazy_init: If True, delay Gmail client initialization until first use.
        """
        self.gmail = gmail_client
        self._owns_gmail_client = gmail_client is None
        self._lazy_init = lazy_init
        if self._owns_gmail_client and not lazy_init:
            self.gmail = get_gmail_client(port=8080)

    def _ensure_gmail_client(self):
        """Ensure Gmail client is initialized (for lazy initialization)."""
        if self.gmail is None and self._owns_gmail_client:
            self.gmail = get_gmail_client(port=8080)

    def strip_html(self, html_content: str) -> str:
        """Remove HTML tags and extract text content. Replace images with size-based placeholders."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Replace images with size-based placeholders before extracting text
        for img in soup.find_all('img'):
            width = img.get('width')
            height = img.get('height')
            alt = img.get('alt', '')
            
            if width and height:
                try:
                    w = int(width)
                    h = int(height)
                    # Categorize by size
                    if w <= 10 and h <= 10:
                        placeholder = "[TRACKING-PIXEL]"
                    elif w >= 500 or h >= 400:
                        placeholder = f"[LARGE-IMAGE {width}x{height}]"
                    else:
                        placeholder = f"[IMAGE {width}x{height}]"
                except ValueError:
                    # Non-numeric dimensions (e.g., "100%")
                    placeholder = f"[IMAGE {width}x{height}]"
            else:
                # No dimensions - use alt text if available
                if alt:
                    placeholder = f"[IMAGE: {alt}]"
                else:
                    placeholder = "[IMAGE]"
            
            img.replace_with(placeholder)
        
        text_content = soup.get_text(separator=" ", strip=True)
        text_content = re.sub(r"\s+", " ", text_content).strip()
        
        # Handle plain text emails with [image: alt] format (from text/plain MIME parts)
        # Convert to our standard format: [IMAGE: alt]
        text_content = re.sub(r'\[image:\s*([^\]]+)\]', r'[IMAGE: \1]', text_content, flags=re.IGNORECASE)
        
        return text_content

    def fetch_emails_from_gmail(
        self, query: str = "is:unread", limit: Optional[int] = None, *, include_body: bool
    ) -> List[Dict[str, Any]]:
        """Fetch emails from Gmail and return their get_email_content() dicts as-is.

        Args:
            query: Gmail search query.
            limit: Maximum number of messages to fetch.
            include_body (required): When False, request Gmail's ``metadata`` format
                (headers only) instead of downloading full message bodies. Used when the
                transform stage classifies from headers alone (llm_body_mode=none),
                saving bandwidth and latency; categorization still works because the
                classification headers are captured either way.

        Returns:
            One get_email_content() dict per fetched email, unchanged: id, subject,
            from, date, headers, has_unsubscribe, and body (present only for full
            downloads). Emails that fail to fetch are logged and skipped, so one
            bad email never aborts the batch.
        """
        self._ensure_gmail_client()
        messages = fetch_emails(self.gmail, query, max_results=limit)

        fetch_kwargs = (
            {"format": "full"}
            if include_body
            else {"format": "metadata", "metadata_headers": CLASSIFICATION_METADATA_HEADERS}
        )

        emails_data = []
        for msg in messages:
            try:
                emails_data.append(get_email_content(self.gmail, msg["id"], **fetch_kwargs))
            except Exception as e:
                logging.error(f"Failed to fetch email {msg['id']}: {e}")
                continue

        return emails_data

    def get_or_create_label(self, label_name: str) -> Optional[str]:
        """Get or create a Gmail label and return its ID."""
        self._ensure_gmail_client()
        return get_or_create_label(self.gmail, label_name)

    def add_labels_to_email(self, email_id: str, label_ids: List[str]) -> bool:
        """Add labels to a specific email."""
        self._ensure_gmail_client()
        return add_labels_to_email(self.gmail, email_id, label_ids)

    def remove_from_inbox(self, email_id: str) -> bool:
        """Remove an email from the inbox."""
        self._ensure_gmail_client()
        return remove_from_inbox(self.gmail, email_id)
