package com.example.wmbservice.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Entity representing an actual budget transaction, mapped to the 'budget_transactions' table.
 * Includes rowHash for deduplication enforcement.
 *
 * Note: Uses Lombok @Getter and @Setter to reduce boilerplate for accessors.
 */
@Getter
@Setter
@Entity
@Table(
        name = "budget_transactions",
        indexes = {
                @Index(name = "idx_statement_period", columnList = "statement_period"),
                @Index(name = "idx_account", columnList = "account"),
                @Index(name = "idx_payment_method", columnList = "payment_method"),
                @Index(name = "idx_category", columnList = "category"),
                @Index(name = "uniq_transaction_hash", columnList = "row_hash, statement_period", unique = true)
        }
)
public class BudgetTransaction {

    private static final Logger logger = LoggerFactory.getLogger(BudgetTransaction.class);

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    @Size(max = 255)
    @Column(nullable = false)
    private String name;

    @NotNull
    @DecimalMin(value = "0.00", inclusive = false)
    @Digits(integer = 12, fraction = 2)
    @Column(nullable = false)
    private BigDecimal amount;

    @NotBlank
    @Size(max = 128)
    @Column(nullable = false)
    private String category;

    @NotBlank
    @Size(max = 32)
    @Column(nullable = false)
    private String criticality;

    @NotNull
    @Column(name = "transaction_date", nullable = false)
    private LocalDate transactionDate;

    @NotBlank
    @Size(max = 32)
    @Column(nullable = false)
    private String account;

    @Size(max = 64)
    private String status;

    private LocalDateTime createdTime;

    @NotBlank
    @Size(max = 64)
    @Column(name = "payment_method", nullable = false)
    private String paymentMethod;

    @NotBlank
    @Size(max = 32)
    @Column(name = "statement_period", nullable = false)
    private String statementPeriod;

    @Size(max = 64)
    @Column(name = "row_hash", length = 64, unique = true)
    private String rowHash;

    /**
     * Default constructor.
     */
    public BudgetTransaction() {
        logger.debug("BudgetTransaction default constructor called");
    }

    /**
     * All-args constructor for entity instantiation.
     */
    public BudgetTransaction(String name, BigDecimal amount, String category, String criticality,
                             LocalDate transactionDate, String account, String status, LocalDateTime createdTime,
                             String paymentMethod, String statementPeriod, String rowHash) {
        logger.info("BudgetTransaction instantiated with name={}, amount={}, statementPeriod={}", name, amount, statementPeriod);
        this.name = name;
        this.amount = amount;
        this.category = category;
        this.criticality = criticality;
        this.transactionDate = transactionDate;
        this.account = account;
        this.status = status;
        this.createdTime = createdTime;
        this.paymentMethod = paymentMethod;
        this.statementPeriod = statementPeriod;
        this.rowHash = rowHash;
    }

    @Override
    public boolean equals(Object o) {
        logger.debug("equals called");
        if (this == o) return true;
        if (!(o instanceof BudgetTransaction that)) return false;
        return Objects.equals(rowHash, that.rowHash) &&
                Objects.equals(statementPeriod, that.statementPeriod);
    }

    @Override
    public int hashCode() {
        logger.debug("hashCode called");
        return Objects.hash(rowHash, statementPeriod);
    }

    @Override
    public String toString() {
        logger.debug("toString called");
        return "BudgetTransaction{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", amount=" + amount +
                ", category='" + category + '\'' +
                ", criticality='" + criticality + '\'' +
                ", transactionDate=" + transactionDate +
                ", account='" + account + '\'' +
                ", status='" + status + '\'' +
                ", createdTime=" + createdTime +
                ", paymentMethod='" + paymentMethod + '\'' +
                ", statementPeriod='" + statementPeriod + '\'' +
                ", rowHash='" + rowHash + '\'' +
                '}';
    }
}