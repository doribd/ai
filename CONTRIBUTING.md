# Contributing

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Follow the [Setup Guide](SETUP.md) to get the project running
4. Create a new branch for your changes

## Development Setup

### Prerequisites

- Java 25 (OpenJDK)
- Docker and Docker Compose
- Maven (or use the included wrapper)
- An OpenAI API key

### Running Locally

```bash
# Start the database
docker-compose up -d

# Build and run
./mvnw spring-boot:run
```

### Running Tests

```bash
./mvnw test
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](../../issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Java version, OS, etc.)

### Suggesting Features

1. Check existing [Issues](../../issues) for similar suggestions
2. Create a new issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Any alternatives you've considered

### Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style below

3. Write or update tests as needed

4. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/output if applicable

## Code Style

### Java

- Follow standard Java naming conventions
- Use meaningful variable and method names
- Add JavaDoc comments for public methods
- Keep methods focused and concise

### Formatting

- Use 4 spaces for indentation (not tabs)
- Maximum line length: 120 characters
- Use blank lines to separate logical blocks

### Example

```java
/**
 * Processes the user query and returns a response.
 *
 * @param message The user's question
 * @return The AI-generated response
 */
public String chat(String message) {
    var documents = retrieveRelevantDocuments(message);
    var prompt = buildPrompt(message, documents);
    return callOpenAI(prompt);
}
```

## Pull Request Guidelines

- Keep PRs focused on a single change
- Update documentation if needed
- Ensure all tests pass
- Respond to review feedback promptly

## Roadmap Items

Current roadmap items that need contributors:

- [ ] Dynamic load of all PDFs from folder
- [ ] Allow multiple queries in a session (currently runs once on startup)

See the [README](README.md) for the full roadmap.

## Questions?

Feel free to open an issue for any questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
