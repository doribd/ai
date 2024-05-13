package com.mytrail.ai;


import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.ExtractedTextFormatter;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This is the main class of the AI application.
 * It is responsible for initializing the application and running the chatbot.
 */
@SpringBootApplication
public class AiApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiApplication.class, args);
    }


    /**
     * Splits the given text into tokens.
     *
     * @return A TokenTextSplitter instance.
     */
    @Bean
    TokenTextSplitter tokenTextSplitter() {
        return new TokenTextSplitter();
    }

    /**
     * Initializes the vector store by deleting existing data and populating it with data from a PDF resource.
     *
     * @param vectorStore The vector store to initialize.
     * @param template    The JdbcTemplate instance.
     * @param pdfResource The PDF resource containing the data.
     * @throws Exception If an error occurs during initialization.
     */
    static void init(VectorStore vectorStore, JdbcTemplate template, Resource pdfResource)
            throws Exception {

        template.update("delete from vector_store");
        var config = PdfDocumentReaderConfig.builder()
                .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder().withNumberOfBottomTextLinesToDelete(3)
                        .withNumberOfTopPagesToSkipBeforeDelete(1)
                        .build())
                .withPagesPerDocument(1)
                .build();

        var pdfReader = new PagePdfDocumentReader(pdfResource, config);
        var textSplitter = new TokenTextSplitter();
        vectorStore.accept(textSplitter.apply(pdfReader.get()));

    }

    /**
     * Creates and returns an instance of {@link ApplicationRunner}.
     *
     * @param chatbot      The {@link Chatbot} instance.
     * @param vectorStore  The {@link VectorStore} instance.
     * @param jdbcTemplate The {@link JdbcTemplate} instance.
     * @param resource     The PDF resource containing the data.
     * @return An instance of {@link ApplicationRunner}.
     */
    @Bean
    ApplicationRunner applicationRunner(
            Chatbot chatbot,
            VectorStore vectorStore,
            JdbcTemplate jdbcTemplate,
            @Value("file:pdfs/Olympic_Games.pdf") Resource resource) {
        return args -> {
            init(vectorStore, jdbcTemplate, resource);
            var response = chatbot.chat("how many rings in the sign and what are colors are they?");
            System.out.println(Map.of("response", response));
        };
    }
}


/**
 * Chatbot class represents an AI chatbot specialized in the Olympic games.
 * It uses a vector store to provide accurate answers based on similarity search to the given message.
 */
@Component
class Chatbot {
    private final String template = """
            You're a wikipedia expert specialized in the olympic games.
            Provide accurate answers but act as if you knew this information innately.
            If unsure, simply state that you don't know.
            DOCUMENTS:
            {documents}
            """;
    private final ChatClient aiClient;
    private final VectorStore vectorStore;

    Chatbot(ChatClient aiClient, VectorStore vectorStore) {
        this.aiClient = aiClient;
        this.vectorStore = vectorStore;
    }

    public String chat(String message) {
        var listOfSimilarDocuments = this.vectorStore.similaritySearch(message);
        var documents = listOfSimilarDocuments
                .stream()
                .map(Document::getContent)
                .collect(Collectors.joining(System.lineSeparator()));
        var systemMessage = new SystemPromptTemplate(this.template)
                .createMessage(Map.of("documents", documents));
        var userMessage = new UserMessage(message);
        var prompt = new Prompt(List.of(systemMessage, userMessage));
        var aiResponse = aiClient.call(prompt);
        return aiResponse.getResult().getOutput().getContent();
    }
}