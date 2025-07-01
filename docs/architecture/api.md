# API Documentation

## Base URL
`https://api.yourdomain.com/v1`

## Authentication
```http
Authorization: Bearer your_api_key
```

## Endpoints

### 1. Chat Completion

#### Generate Text
```http
POST /chat/completions
```

**Request Body**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "concept_controls": {
    "toxicity": {
      "strength": 0.5,
      "direction": "minimize"
    }
  },
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt2",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "concept_analysis": {
        "toxicity": 0.1,
        "safety": 0.9
      }
    },
    "finish_reason": "stop"
  }]
}
```

### 2. Concept Analysis

#### Analyze Text
```http
POST /concepts/analyze
```

**Request Body**
```json
{
  "text": "This is a sample text to analyze",
  "concepts": ["toxicity", "safety"]
}
```

**Response**
```json
{
  "analysis": {
    "toxicity": 0.85,
    "safety": 0.2,
    "explanation": "The text shows high toxicity and low safety scores."
  }
}
```

### 3. Model Management

#### List Available Models
```http
GET /models
```

**Response**
```json
{
  "models": [
    {
      "id": "gpt2",
      "name": "GPT-2 Small",
      "description": "Small version of GPT-2",
      "capabilities": ["text-generation", "concept-detection"]
    }
  ]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": {
    "message": "Invalid request format",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "invalid_parameter"
  }
}
```

### 401 Unauthorized
```json
{
  "error": {
    "message": "Invalid API key",
    "type": "authentication_error",
    "code": "invalid_api_key"
  }
}
```

## Rate Limits
- 60 requests per minute per API key
- 100,000 tokens per minute
