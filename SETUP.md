# Setup Guide

This guide will walk you through setting up and running the Spring AI RAG Chatbot.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Java 25** (OpenJDK recommended)
- **Docker** and **Docker Compose**
- **Maven** (or use the included Maven wrapper)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

## Step 1: Clone the Repository

```bash
git clone https://github.com/doribd/spring-ai.git
cd spring-ai
```

## Step 2: Start the Database

Start PostgreSQL with PGVector extension using Docker Compose:

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** with PGVector on port `5432`
- **PgAdmin** web interface on port `5050`

Verify the containers are running:

```bash
docker-compose ps
```

## Step 3: Configure OpenAI API Key

Edit `src/main/resources/application.properties` and replace the placeholder with your API key:

```properties
spring.ai.openai.api-key=sk-your-actual-api-key-here
```

Alternatively, set it as an environment variable:

```bash
export SPRING_AI_OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Step 4: Add Your PDF (Optional)

The project comes with a sample PDF about the Olympic Games. To use your own PDF:

1. Place your PDF file in the `pdfs/` folder
2. Update the resource path in `AiApplication.java`:

```java
@Value("file:pdfs/YourDocument.pdf") Resource resource
```

## Step 5: Build the Application

Using Maven wrapper (recommended):

```bash
./mvnw clean package -DskipTests
```

Or with Maven installed globally:

```bash
mvn clean package -DskipTests
```

## Step 6: Run the Application

```bash
./mvnw spring-boot:run
```

Or run the JAR directly:

```bash
java -jar target/ai-1.0.0.jar
```

## Expected Output

On successful startup, you'll see:
1. The PDF being processed and split into chunks
2. Vector embeddings being stored in PostgreSQL
3. A demo query response about Olympic rings:

```
{response=The Olympic symbol consists of five interlocking rings...}
```

## Accessing PgAdmin (Optional)

To view the vector store data:

1. Open http://localhost:5050
2. Login with:
   - Email: `pgadmin4@pgadmin.org`
   - Password: `admin`
3. Connect to the PostgreSQL server:
   - Host: `postgres`
   - Port: `5432`
   - Username: `postgres`
   - Password: `postgres`
   - Database: `vector_store`

## Troubleshooting

### Connection refused to PostgreSQL

Ensure Docker containers are running:
```bash
docker-compose up -d
docker-compose logs postgres
```

### OpenAI API errors

- Verify your API key is correct
- Check your OpenAI account has available credits at [platform.openai.com/usage](https://platform.openai.com/usage)

### PDF not found

Ensure the PDF path is correct and the file exists in the `pdfs/` folder.

### Java version issues

Verify you're using Java 25:
```bash
java -version
```

## Next Steps

- See [ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand how RAG works
- See [CONFIGURATION.md](docs/CONFIGURATION.md) for all configuration options
- Check the [README.md](README.md) for the project roadmap
