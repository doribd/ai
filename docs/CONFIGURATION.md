# Configuration Guide

This document describes all configuration options for the Spring AI RAG Chatbot.

## Application Properties

Configuration is managed in `src/main/resources/application.properties`.

### Database Configuration

| Property | Description | Default |
|----------|-------------|---------|
| `spring.datasource.url` | PostgreSQL connection URL | `jdbc:postgresql://localhost/vector_store` |
| `spring.datasource.username` | Database username | `postgres` |
| `spring.datasource.password` | Database password | `postgres` |

Example:
```properties
spring.datasource.url=jdbc:postgresql://localhost/vector_store
spring.datasource.username=postgres
spring.datasource.password=postgres
```

### OpenAI Configuration

| Property | Description | Required |
|----------|-------------|----------|
| `spring.ai.openai.api-key` | Your OpenAI API key | Yes |
| `spring.ai.openai.chat.model` | Chat model to use | No (defaults to gpt-3.5-turbo) |
| `spring.ai.openai.embedding.model` | Embedding model | No (defaults to text-embedding-ada-002) |

Example:
```properties
spring.ai.openai.api-key=sk-your-api-key-here
spring.ai.openai.chat.model=gpt-4
spring.ai.openai.embedding.model=text-embedding-3-small
```

### Environment Variables

All properties can be overridden via environment variables using Spring Boot's relaxed binding:

| Environment Variable | Property |
|---------------------|----------|
| `SPRING_DATASOURCE_URL` | `spring.datasource.url` |
| `SPRING_DATASOURCE_USERNAME` | `spring.datasource.username` |
| `SPRING_DATASOURCE_PASSWORD` | `spring.datasource.password` |
| `SPRING_AI_OPENAI_API_KEY` | `spring.ai.openai.api-key` |

Example:
```bash
export SPRING_AI_OPENAI_API_KEY=sk-your-api-key-here
./mvnw spring-boot:run
```

## Docker Compose Configuration

The `docker-compose.yml` file configures the database services.

### PostgreSQL (PGVector)

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database superuser | `postgres` |
| `POSTGRES_PASSWORD` | Superuser password | `postgres` |
| `POSTGRES_DB` | Default database name | `vector_store` |

### PgAdmin

| Variable | Description | Default |
|----------|-------------|---------|
| `PGADMIN_DEFAULT_EMAIL` | Admin login email | `pgadmin4@pgadmin.org` |
| `PGADMIN_DEFAULT_PASSWORD` | Admin password | `admin` |
| `PGADMIN_PORT` | Web interface port | `5050` |

Override defaults:
```bash
PGADMIN_DEFAULT_PASSWORD=secure_password docker-compose up -d
```

## PDF Configuration

The PDF source is configured in `AiApplication.java`:

```java
@Value("file:pdfs/Olympic_Games.pdf") Resource resource
```

### PDF Reader Options

Configured programmatically in the `init` method:

```java
var config = PdfDocumentReaderConfig.builder()
    .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder()
        .withNumberOfBottomTextLinesToDelete(3)  // Skip footer lines
        .withNumberOfTopPagesToSkipBeforeDelete(1)  // Skip first page headers
        .build())
    .withPagesPerDocument(1)  // One page per document chunk
    .build();
```

| Option | Description |
|--------|-------------|
| `numberOfBottomTextLinesToDelete` | Lines to remove from page bottom (footers) |
| `numberOfTopPagesToSkipBeforeDelete` | Pages to skip before applying deletions |
| `pagesPerDocument` | How many pages per document chunk |

## Vector Store Configuration

Spring AI auto-configures the PGVector store. Additional options:

```properties
# Embedding dimensions (must match your embedding model)
spring.ai.vectorstore.pgvector.dimensions=1536

# Distance type for similarity search
spring.ai.vectorstore.pgvector.distance-type=COSINE_DISTANCE

# Index type
spring.ai.vectorstore.pgvector.index-type=HNSW
```

## Advanced Configuration

### Custom Chat Model

To use GPT-4 or other models:

```properties
spring.ai.openai.chat.model=gpt-4
spring.ai.openai.chat.temperature=0.7
spring.ai.openai.chat.max-tokens=1000
```

### Connection Pool

For production, configure HikariCP:

```properties
spring.datasource.hikari.maximum-pool-size=10
spring.datasource.hikari.minimum-idle=5
spring.datasource.hikari.connection-timeout=30000
```

### Logging

Enable debug logging for troubleshooting:

```properties
logging.level.org.springframework.ai=DEBUG
logging.level.com.mytrail.ai=DEBUG
```

## Profiles

Create environment-specific configurations:

**application-dev.properties:**
```properties
spring.ai.openai.chat.model=gpt-3.5-turbo
logging.level.root=DEBUG
```

**application-prod.properties:**
```properties
spring.ai.openai.chat.model=gpt-4
logging.level.root=INFO
```

Activate a profile:
```bash
./mvnw spring-boot:run -Dspring-boot.run.profiles=dev
```
