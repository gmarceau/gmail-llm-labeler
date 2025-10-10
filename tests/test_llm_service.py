"""Tests for LLMService class."""

import json
from unittest.mock import MagicMock, patch

import pytest

from email_labeler.llm_service import LLMCategorizationError, LLMService

# Test categories for all tests
TEST_CATEGORIES = ["Marketing", "Work", "Personal", "Bills", "Newsletters", "Other"]
TEST_MAX_CONTENT_LENGTH = 2000


class TestLLMService:
    """Test cases for LLMService class."""

    def test_init_with_client(self, mock_openai_client):
        """Test initialization with provided client."""
        model = "gpt-4"
        service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model=model,
        )

        assert service.llm_client == mock_openai_client
        assert service.model == model
        assert service.categories == TEST_CATEGORIES
        assert service.max_content_length == TEST_MAX_CONTENT_LENGTH

    def test_init_without_client_openai(self):
        """Test initialization without client for OpenAI."""
        with patch("email_labeler.llm_service.LLM_SERVICE", "OpenAI"), patch(
            "email_labeler.llm_service.OPENAI_API_KEY", "test-key"
        ), patch("email_labeler.llm_service.OPENAI_MODEL", "gpt-3.5-turbo"), patch(
            "email_labeler.llm_service.OpenAI"
        ) as mock_openai:
            service = LLMService(
                categories=TEST_CATEGORIES, max_content_length=TEST_MAX_CONTENT_LENGTH
            )

            mock_openai.assert_called_once_with(api_key="test-key")
            assert service.model == "gpt-3.5-turbo"

    def test_init_without_client_ollama(self):
        """Test initialization without client for Ollama."""
        with patch("email_labeler.llm_service.LLM_SERVICE", "Ollama"), patch(
            "email_labeler.llm_service.OLLAMA_BASE_URL", "http://localhost:11434/v1"
        ), patch("email_labeler.llm_service.OLLAMA_MODEL", "llama2"), patch(
            "email_labeler.llm_service.OpenAI"
        ) as mock_openai:
            service = LLMService(
                categories=TEST_CATEGORIES, max_content_length=TEST_MAX_CONTENT_LENGTH
            )

            mock_openai.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )
            assert service.model == "llama2"

    def test_categorize_email_success(self, real_llm_service, mock_openai_client):
        """Test successful email categorization."""
        email_content = "Meeting tomorrow at 2 PM with the development team."
        expected_response = {"category": "Work", "explanation": "Business meeting notification"}

        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(expected_response)

        category, explanation = real_llm_service.categorize_email(email_content)

        assert category == "Work"
        assert explanation == "Business meeting notification"

        # Verify the API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert "messages" in call_args[1]

    def test_categorize_email_truncation(self, mock_openai_client):
        """Test email content truncation for long emails."""
        long_content = "A" * 3000  # Long content to test truncation
        expected_response = {"category": "Other", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(expected_response)

        # Create service with custom max_content_length
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=2000,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )
        llm_service.categorize_email(long_content)

        call_args = mock_openai_client.chat.completions.create.call_args
        content_in_prompt = call_args[1]["messages"][1]["content"]
        # Should contain truncated content marker
        assert "[Email truncated for processing]" in content_in_prompt

    def test_categorize_email_invalid_json_response(self, real_llm_service, mock_openai_client):
        """Test handling invalid JSON response from LLM."""
        email_content = "Test email content"
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Invalid JSON"

        category, explanation = real_llm_service.categorize_email(email_content)

        assert category == "Other"
        assert explanation == "Failed to parse response"

    def test_categorize_email_missing_fields(self, mock_openai_client):
        """Test handling response with missing required fields."""
        email_content = "Test email content"
        incomplete_response = {"category": "Work"}  # Missing explanation

        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(incomplete_response)

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        category, explanation = llm_service.categorize_email(email_content)

        assert category == "Work"
        assert explanation == ""  # Default empty explanation

    def test_categorize_email_api_error(self, mock_openai_client):
        """Test handling API errors during categorization."""
        email_content = "Test email content"
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        with pytest.raises(LLMCategorizationError) as exc_info:
            llm_service.categorize_email(email_content)

        assert "LLM categorization failed" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)

    def test_categorize_email_invalid_category(self, mock_openai_client):
        """Test handling of invalid category in response."""
        email_content = "Test email content"
        response = {"category": "InvalidCategory", "explanation": "Test explanation"}

        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(response)

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        category, explanation = llm_service.categorize_email(email_content)

        assert category == "Other"
        assert "Unknown category: InvalidCategory" in explanation

    def test_get_llm_client_openai(self):
        """Test getting OpenAI client."""
        with patch("email_labeler.llm_service.LLM_SERVICE", "OpenAI"), patch(
            "email_labeler.llm_service.OPENAI_API_KEY", "test-key"
        ), patch("email_labeler.llm_service.OpenAI") as mock_openai:
            service = LLMService(
                categories=TEST_CATEGORIES, max_content_length=TEST_MAX_CONTENT_LENGTH
            )
            service._get_llm_client()

            mock_openai.assert_called_with(api_key="test-key")

    def test_get_llm_client_ollama(self):
        """Test getting Ollama client."""
        with patch("email_labeler.llm_service.LLM_SERVICE", "Ollama"), patch(
            "email_labeler.llm_service.OLLAMA_BASE_URL", "http://localhost:11434/v1"
        ), patch("email_labeler.llm_service.OpenAI") as mock_openai:
            service = LLMService(
                categories=TEST_CATEGORIES, max_content_length=TEST_MAX_CONTENT_LENGTH
            )
            service._get_llm_client()

            mock_openai.assert_called_with(base_url="http://localhost:11434/v1", api_key="ollama")

    @pytest.mark.parametrize(
        "content,expected_category",
        [
            ("Meeting with client tomorrow", "Work"),
            ("Your subscription has been renewed", "Newsletters"),
            ("Lunch plans this weekend", "Personal"),
            ("Account balance notification", "Bills"),
        ],
    )
    def test_categorize_various_emails(self, mock_openai_client, content, expected_category):
        """Test categorizing various types of emails."""
        response = {"category": expected_category, "explanation": "Test explanation"}
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(response)

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        category, explanation = llm_service.categorize_email(content)

        assert category == expected_category
        assert explanation == "Test explanation"

    def test_categorize_email_with_reasoning(self, mock_openai_client):
        """Test categorization with detailed reasoning enabled."""
        email_content = "Project deadline reminder"
        expected_response = {"category": "Work", "explanation": "Business related"}

        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(expected_response)

        with patch("email_labeler.llm_service.LLM_SERVICE", "Ollama"), patch(
            "email_labeler.llm_service.GPT_OSS_REASONING", "medium"
        ):
            # Create LLMService with mocked client - use a gpt-oss model to trigger reasoning
            llm_service = LLMService(
                categories=TEST_CATEGORIES,
                max_content_length=TEST_MAX_CONTENT_LENGTH,
                llm_client=mock_openai_client,
                model="gpt-oss-instruct",
            )

            llm_service.categorize_email(email_content)

            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]

            # Should include reasoning in the prompt
            system_message = messages[0]["content"]
            assert "Reasoning:" in system_message

    def test_categorize_empty_email(self, mock_openai_client):
        """Test categorizing empty email content."""
        empty_content = ""
        response = {"category": "Other", "explanation": "Empty email"}
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(response)

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        category, explanation = llm_service.categorize_email(empty_content)

        assert category == "Other"
        assert explanation == "Empty email"

    def test_log_interaction(self, mock_openai_client):
        """Test logging of LLM interactions."""
        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        with patch("builtins.open", create=True) as mock_open, patch(
            "email_labeler.llm_service.LLM_LOG_FILE", "/tmp/test.log"
        ):
            # Test the _log_interaction method
            llm_service._log_interaction(1.0, 2.5, '{"category": "Work"}')

            mock_open.assert_called_once_with("/tmp/test.log", "a")

    def test_different_models(self):
        """Test service with different models."""
        models = ["gpt-3.5-turbo", "gpt-4", "llama2"]

        for model in models:
            mock_client = MagicMock()
            service = LLMService(
                categories=TEST_CATEGORIES,
                max_content_length=TEST_MAX_CONTENT_LENGTH,
                llm_client=mock_client,
                model=model,
            )
            assert service.model == model

    def test_timeout_handling(self, mock_openai_client):
        """Test handling of API timeouts."""
        from openai import APITimeoutError

        email_content = "Test email"
        mock_openai_client.chat.completions.create.side_effect = APITimeoutError("Request Timeout")

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        with pytest.raises(LLMCategorizationError) as exc_info:
            llm_service.categorize_email(email_content)

        assert "LLM categorization failed" in str(exc_info.value)

    def test_rate_limit_handling(self, mock_openai_client):
        """Test handling of rate limit errors."""
        from openai import RateLimitError

        email_content = "Test email"
        mock_openai_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=MagicMock(), body={}
        )

        # Create LLMService with mocked client
        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        with pytest.raises(LLMCategorizationError) as exc_info:
            llm_service.categorize_email(email_content)

        assert "LLM categorization failed" in str(exc_info.value)

    def test_unsupported_service_fallback(self):
        """Test handling of unsupported service configurations."""
        # The actual implementation doesn't validate service type, it just defaults to OpenAI
        with patch("email_labeler.llm_service.LLM_SERVICE", "UnsupportedService"), patch(
            "email_labeler.llm_service.OPENAI_API_KEY", "test-key"
        ), patch("email_labeler.llm_service.OpenAI") as mock_openai:
            LLMService(categories=TEST_CATEGORIES, max_content_length=TEST_MAX_CONTENT_LENGTH)
            # Should fall through to OpenAI case since it's not "Ollama"
            mock_openai.assert_called_with(api_key="test-key")

    def test_custom_system_prompt(self, mock_openai_client):
        """Test using a custom system prompt."""
        custom_system = "You are a specialized email classifier."
        email_content = "Test email"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
            system_prompt=custom_system,
        )

        llm_service.categorize_email(email_content)

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_system

    def test_custom_user_prompt(self, mock_openai_client):
        """Test using a custom user prompt with template variables."""
        custom_user = "Classify: {email_content}. Categories: {categories}"
        email_content = "Test email"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
            user_prompt=custom_user,
        )

        llm_service.categorize_email(email_content)

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[1]["role"] == "user"
        # Check that template variables were substituted
        assert "Test email" in messages[1]["content"]
        assert "Marketing" in messages[1]["content"]

    def test_template_variable_substitution(self, mock_openai_client):
        """Test that template variables are properly substituted."""
        custom_user = "Email: {email_content}\nCategories: {categories}"
        email_content = "Meeting tomorrow"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        llm_service = LLMService(
            categories=["Work", "Personal"],
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
            user_prompt=custom_user,
        )

        llm_service.categorize_email(email_content)

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]

        assert "Email: Meeting tomorrow" in user_message
        assert "Categories: Work, Personal" in user_message

    def test_custom_prompts_both(self, mock_openai_client):
        """Test using both custom system and user prompts."""
        custom_system = "You are a test assistant."
        custom_user = "Process: {email_content}"
        email_content = "Test email"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
            system_prompt=custom_system,
            user_prompt=custom_user,
        )

        llm_service.categorize_email(email_content)

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        assert messages[0]["content"] == custom_system
        assert "Process: Test email" in messages[1]["content"]

    def test_invalid_template_variable(self, mock_openai_client):
        """Test handling of invalid template variables."""
        custom_user = "Content: {email_content}, Invalid: {nonexistent_var}"
        email_content = "Test email"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
            user_prompt=custom_user,
        )

        # Should not raise an error, but log a warning
        category, explanation = llm_service.categorize_email(email_content)

        assert category == "Work"
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        # Invalid variable should be replaced with empty string
        assert "Invalid: " in messages[1]["content"]

    def test_default_prompts_still_work(self, mock_openai_client):
        """Test that default prompts still work when no custom prompts provided."""
        email_content = "Test email"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        llm_service = LLMService(
            categories=TEST_CATEGORIES,
            max_content_length=TEST_MAX_CONTENT_LENGTH,
            llm_client=mock_openai_client,
            model="gpt-3.5-turbo",
        )

        category, explanation = llm_service.categorize_email(email_content)

        assert category == "Work"
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        # Check default prompts are used
        assert "email categorization assistant" in messages[0]["content"].lower()
        assert "Categorize this email" in messages[1]["content"]

    def test_reasoning_template_variable(self, mock_openai_client):
        """Test that reasoning template variable works for GPT-OSS models."""
        custom_system = "Reasoning: {reasoning}\nYou are an assistant."
        email_content = "Test email"
        response = {"category": "Work", "explanation": "Test"}

        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(response)
        )

        with patch("email_labeler.llm_service.GPT_OSS_REASONING", "high"):
            llm_service = LLMService(
                categories=TEST_CATEGORIES,
                max_content_length=TEST_MAX_CONTENT_LENGTH,
                llm_client=mock_openai_client,
                model="gpt-oss-instruct",
                system_prompt=custom_system,
            )

            llm_service.categorize_email(email_content)

            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]

            assert "Reasoning: high" in messages[0]["content"]
