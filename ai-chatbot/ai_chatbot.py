import os
import logging
import traceback
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "huggingface").lower()

# Global cache for lazy-loaded models
_cached_models = {}


class ModelProvider:
    """Base class for model providers."""
    
    def __init__(self):
        """Initialize the base model provider with a generic name."""
        self.name = "Unknown"
    
    def query(self, prompt: str) -> str:
        """
        Query the model and return response.
        
        Args:
            prompt (str): The input text to send to the model.
            
        Returns:
            str: The model's response text.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI GPT model provider."""
    
    def __init__(self):
        """
        Initialize the OpenAI provider with ChatOpenAI client.
        
        Raises:
            ImportError: If langchain-openai package is not installed.
            ValueError: If OPENAI_API_KEY environment variable is not set.
        """
        super().__init__()
        self.name = "OpenAI"
        try:
            from langchain_openai import ChatOpenAI
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            
            self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key)
        except ImportError:
            raise ImportError("langchain-openai package required. Install with: pip install langchain-openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise
    
    def query(self, prompt: str) -> str:
        """
        Query the OpenAI model with the provided prompt.
        
        Args:
            prompt (str): The input text to send to OpenAI.
            
        Returns:
            str: The generated response from OpenAI, or an error message if the query fails.
        """
        try:
            response = self.llm.invoke(prompt)
            # Handle both response objects and strings
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"OpenAI query error: {e}\n{traceback.format_exc()}")
            return f"Error calling OpenAI: {e}"


class GeminiProvider(ModelProvider):
    """Google Gemini model provider."""
    
    def __init__(self):
        """
        Initialize the Google Gemini provider.
        
        Raises:
            ImportError: If google-generativeai package is not installed.
            ValueError: If GOOGLE_API_KEY environment variable is not set.
        """
        super().__init__()
        self.name = "Google Gemini"
        try:
            import google.generativeai as genai
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise
    
    def query(self, prompt: str) -> str:
        """
        Query the Google Gemini model with the provided prompt.
        
        Args:
            prompt (str): The input text to send to Gemini.
            
        Returns:
            str: The generated response from Gemini, or an error message if the query fails.
        """
        try:
            response = self.model.generate_content(prompt)
            # Handle both response objects and strings
            if hasattr(response, "text"):
                return response.text
            return str(response)
        except Exception as e:
            logger.error(f"Gemini query error: {e}\n{traceback.format_exc()}")
            return f"Error calling Gemini: {e}"


class HuggingFaceProvider(ModelProvider):
    """
    HuggingFaceProvider - Hugging Face Model Provider Module

    OVERVIEW:
    This module implements a ModelProvider subclass that enables interaction with Hugging Face language models.
    It supports two execution modes: Local model execution and Cloud API-based execution.

    EXECUTION MODES:
    1. LOCAL MODE: Downloads and runs models directly on your machine (requires GPU/CPU resources)
    2. API MODE: Sends requests to Hugging Face Inference API (requires internet and API token)

    SYSTEMATIC WORKFLOW STEPS:

    STEP 1 - INITIALIZATION (__init__)
        Purpose: Set up the Hugging Face provider with configuration
        Dependencies: Environment variables (USE_LOCAL_MODEL, HF_CHAT_MODEL, etc.)
        Actions:
            a) Determine execution mode (local vs API) from USE_LOCAL_MODEL env var
            b) Set model name from HF_CHAT_MODEL (defaults to Mistral-7B-Instruct-v0.2)
            c) Configure device (GPU/CPU) for local execution
            d) Set memory optimization flags (8-bit loading)
            e) Initialize local pipeline if local mode is enabled

    STEP 2 - LOCAL PIPELINE INITIALIZATION (_init_local_pipeline)
        Purpose: Download and prepare the model for local execution
        Prerequisites: transformers library, sufficient disk/RAM space
        Actions:
            a) Import transformers.pipeline
            b) Configure pipeline parameters (model name, device, trust_remote_code)
            c) Apply memory optimizations (8-bit loading if enabled)
            d) Load Hugging Face authentication token if needed
            e) Create and cache the pipeline for reuse
        Error Handling: Catches and logs initialization failures

    STEP 3 - QUERY ROUTING (query)
        Purpose: Route user queries to appropriate execution mode
        Input: prompt (str) - User's input text
        Actions:
            a) Check if local mode is enabled
            b) Route to _query_local() if local mode, else _query_api()
            c) Handle exceptions gracefully with error logging
        Output: Generated text response or error message

    STEP 4A - LOCAL QUERY EXECUTION (_query_local)
        Purpose: Generate responses using local model
        Prerequisites: Pipeline must be initialized (auto-initializes if None)
        Actions:
            a) Check and initialize pipeline if not already loaded
            b) Call pipeline with configured parameters:
                - max_new_tokens: Limit response length
                - do_sample: Enable probabilistic generation
                - temperature: Control randomness (0.7 = moderate)
                - top_p: Nucleus sampling threshold (95% most likely tokens)
            c) Parse response object (may be list, dict, or string)
            d) Extract generated text from response structure
            e) Remove echoed prompt from output if present
            f) Return cleaned text response
        Performance: Runs entirely on local machine (no network latency)

    STEP 4B - API QUERY EXECUTION (_query_api)
        Purpose: Generate responses using Hugging Face Inference API
        Prerequisites: HF_API_TOKEN environment variable set
        Actions:
            a) Import InferenceClient from huggingface_hub
            b) Initialize client with API token (or use free tier)
            c) Send chat completion request to specified model
            d) Handle multiple response format variations:
                - Check for "choices[0].message.content" structure
                - Check for "generated_text" field
                - Check for "text" field
                - Fall back to string conversion
            e) Return extracted response content
        Network: Requires internet connection, subject to API rate limits

    CONFIGURATION ENVIRONMENT VARIABLES:
        - USE_LOCAL_MODEL: "true"/"false" - Enable local execution (default: false)
        - HF_CHAT_MODEL: Model identifier (default: mistralai/Mistral-7B-Instruct-v0.2)
        - HF_API_TOKEN: API token for cloud API mode
        - HF_TOKEN: Alternative token variable for gated models
        - HF_DEVICE: "auto"/"cuda"/"cpu" - Device for local execution (default: auto)
        - HF_MAX_TOKENS: Maximum output tokens (default: 512)
        - HF_USE_8BIT: "true"/"false" - Memory optimization (default: false)

    LEARNING NOTES:
        - 8-bit loading reduces memory usage by 75% but may slightly reduce performance
        - Local mode has zero latency after initial setup but requires resources
        - API mode is lightweight but subject to rate limits and costs
        - The pipeline automatically strips echoed prompts from responses
        - Device="auto" automatically selects GPU if available, fallback to CPU
    """

    def __init__(self):
        """
        Initialize the Hugging Face provider with local or API mode configuration.
        
        Reads environment variables to determine execution mode and model configuration.
        For local mode, attempts to initialize the transformers pipeline.
        """
        super().__init__()
        self.name = "Hugging Face"
        self.use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() in ("1", "true", "yes")
        self.model_name = os.getenv("HF_CHAT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.api_token = os.getenv("HF_API_TOKEN")
        self.pipeline = None

        # Local model configuration
        self.device = os.getenv("HF_DEVICE", "auto")  # "auto", "cuda", "cpu"
        self.max_tokens = int(os.getenv("HF_MAX_TOKENS", "512"))
        self.use_8bit = os.getenv("HF_USE_8BIT", "false").lower() in ("1", "true", "yes")

        if self.use_local:
            self.name = "Hugging Face (Local)"
            logger.info(f"Configured for local model execution: {self.model_name}")
            logger.info(f"  Device: {self.device}, 8-bit: {self.use_8bit}, Max tokens: {self.max_tokens}")
            self._init_local_pipeline()
        else:
            self.name = "Hugging Face (API)"
            logger.info(f"Configured for API execution: {self.model_name}")

    def _init_local_pipeline(self):
        """
        Initialize local transformers pipeline with memory optimizations.
        
        Downloads the model specified in HF_CHAT_MODEL and creates a text-generation
        pipeline with optional 8-bit quantization and device mapping.
        
        Raises:
            Exception: If model download or pipeline initialization fails.
        """
        try:
            from transformers import pipeline
            logger.info(f"Loading local model: {self.model_name}")

            # Load model with optimization flags
            pipeline_kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "device_map": self.device,
            }

            # Add 8-bit loading for memory efficiency on local models
            if self.use_8bit:
                pipeline_kwargs["load_in_8bit"] = True
                logger.info("Using 8-bit model loading for reduced memory usage")

            # Use HF token for gated models
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
            if hf_token:
                pipeline_kwargs["token"] = hf_token
                logger.info("Using Hugging Face token for authentication")

            self.pipeline = pipeline("text-generation", **pipeline_kwargs)
            logger.info(f"Successfully loaded local model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load local pipeline: {e}\n{traceback.format_exc()}")
            raise

    def query(self, prompt: str) -> str:
        """
        Query the Hugging Face model using either local or API execution.
        
        Args:
            prompt (str): The input text to send to the model.
            
        Returns:
            str: The generated response, or an error message if the query fails.
        """
        try:
            if self.use_local:
                return self._query_local(prompt)
            else:
                return self._query_api(prompt)
        except Exception as e:
            logger.error(f"Hugging Face query error: {e}\n{traceback.format_exc()}")
            return f"Error calling Hugging Face: {e}"

    def _query_local(self, prompt: str) -> str:
        """
        Query the local transformers pipeline.
        
        Args:
            prompt (str): The input text to generate from.
            
        Returns:
            str: The generated text response, stripped of the echoed prompt.
            
        Raises:
            Exception: If pipeline execution fails.
        """
        if self.pipeline is None:
            self._init_local_pipeline()

        try:
            # Generate text with configured parameters
            resp = self.pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

            # The pipeline may return a list of dicts or a single dict/object
            if isinstance(resp, list) and resp:
                first = resp[0]
                if isinstance(first, dict):
                    text = first.get("generated_text") or first.get("text")
                    if text:
                        # strip echoed prompt if present
                        if isinstance(text, str) and text.startswith(prompt):
                            text = text[len(prompt):].strip()
                        return text
            elif isinstance(resp, dict):
                text = resp.get("generated_text") or resp.get("text")
                if text:
                    if isinstance(text, str) and text.startswith(prompt):
                        text = text[len(prompt):].strip()
                    return text
            # fallback to string conversion
            return str(resp)
        except Exception as e:
            logger.error(f"Local model query failed: {e}\n{traceback.format_exc()}")
            raise

    def _query_api(self, prompt: str) -> str:
        """
        Query the Hugging Face Inference API.
        
        Args:
            prompt (str): The input text to send to the model.
            
        Returns:
            str: The generated response text.
            
        Raises:
            Exception: If API communication fails.
        """
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=self.api_token) if self.api_token else InferenceClient()
            logger.info(f"Querying HF API with model: {self.model_name}")
            resp = client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            # Handle different response formats
            if isinstance(resp, str):
                return resp
            if isinstance(resp, dict):
                # Try multiple possible response structures
                if "choices" in resp and resp["choices"]:
                    choice = resp["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                if "generated_text" in resp:
                    return resp["generated_text"]
                if "text" in resp:
                    return resp["text"]
            # Fallback for object-type responses
            if hasattr(resp, "choices") and resp.choices:
                if hasattr(resp.choices[0], "message") and hasattr(resp.choices[0].message, "content"):
                    return resp.choices[0].message.content
            return str(resp)
        except Exception as e:
            logger.error(f"HF Inference API error: {e}\n{traceback.format_exc()}")
            raise


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""
    
    def __init__(self):
        """
        Initialize the Ollama provider and test connection to Ollama server.
        
        Raises:
            ImportError: If ollama package is not installed.
            ConnectionError: If unable to connect to Ollama server.
        """
        super().__init__()
        self.name = "Ollama (Local)"
        try:
            import ollama
            self.ollama = ollama
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model_name = os.getenv("OLLAMA_MODEL", "llama2")
            
            # Test connection
            self._test_connection()
        except ImportError:
            raise ImportError("ollama package required. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise
    
    def _test_connection(self):
        """
        Test the connection to the Ollama server.
        
        Raises:
            ConnectionError: If the Ollama server is not responding or returns an error status.
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not responding. Status: {response.status_code}")
            logger.info(f"Connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def query(self, prompt: str) -> str:
        """
        Query the Ollama model with the provided prompt.
        
        Args:
            prompt (str): The input text to send to the model.
            
        Returns:
            str: The generated response, or an error message if the query fails.
        """
        try:
            response = self.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
            )
            # Handle both dict and object responses from ollama
            if isinstance(response, dict):
                return response.get("response", "")
            elif hasattr(response, "response"):
                return response.response
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Ollama query error: {e}\n{traceback.format_exc()}")
            return f"Error calling Ollama: {e}"


def get_model_provider() -> ModelProvider:
    """
    Factory function to get and cache the appropriate model provider.
    
    Returns:
        ModelProvider: An initialized provider instance (OpenAI, Gemini, HuggingFace, or Ollama).
        
    Raises:
        ValueError: If MODEL_TYPE environment variable is not recognized.
        Exception: If provider initialization fails.
    """
    global _cached_models
    
    if MODEL_TYPE in _cached_models:
        return _cached_models[MODEL_TYPE]
    
    try:
        if MODEL_TYPE == "openai":
            provider = OpenAIProvider()
        elif MODEL_TYPE == "gemini":
            provider = GeminiProvider()
        elif MODEL_TYPE == "huggingface":
            provider = HuggingFaceProvider()
        elif MODEL_TYPE == "ollama":
            provider = OllamaProvider()
        else:
            raise ValueError(
                f"Unknown MODEL_TYPE: {MODEL_TYPE}. "
                "Must be one of: openai, gemini, huggingface, ollama"
            )
        
        _cached_models[MODEL_TYPE] = provider
        logger.info(f"Initialized model provider: {provider.name}")
        return provider
    except Exception as e:
        error_msg = f"Failed to initialize model provider: {e}"
        logger.error(error_msg)
        raise


def query_model(prompt: str) -> str:
    """
    Query the model using the configured provider.
    
    Args:
        prompt (str): The input text to query the model with.
        
    Returns:
        str: The model's response, or an error message if initialization or query fails.
    """
    try:
        provider = get_model_provider()
        return provider.query(prompt)
    except Exception as e:
        return f"Error: Failed to initialize model. {e}"


def main():
    """
    Main Streamlit application entry point.
    
    Sets up the UI, initializes the model provider, and handles user input
    in a chat-like interface with session history.
    """
    st.title("🤖 AI Powered Chatbot")
    
    try:
        provider = get_model_provider()
        model_info = f"Using {provider.name}"
        st.success(f"✓ {model_info}")
        st.write(f"Chat with an AI model powered by {provider.name}.")
    except Exception as e:
        st.error(f"Failed to load model provider: {e}")
        st.info("Please check your .env configuration and ensure all required credentials are set.")
        return
    
    if "history" not in st.session_state:
        st.session_state.history = []

    # create a container for history that we can refresh
    history_container = st.container()
    with history_container:
        for role, text in st.session_state.history:
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Bot:** {text}")

    # input form placed below history
    with st.form("input_form", clear_on_submit=True):
        user_input = st.text_input("You:")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        st.session_state.history.append(("user", user_input))
        with st.spinner("Generating response..."):
            response = query_model(user_input)
        
        st.session_state.history.append(("bot", response))
        
        # re-render history in the same container so new messages appear above form
        with history_container:
            history_container.empty()
            for role, text in st.session_state.history:
                if role == "user":
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**Bot:** {text}")


if __name__ == "__main__":
    main()