#ifndef LOG_H_
#define LOG_H_

#include <format>
#include <iostream>
#include <chrono>

/// @brief A Singleton class for logging to ensure in the destructor that the
/// buffer that is logged to is flushed before program exit, instead of flushing
/// after every log. Provides methods for logging with pretty timestamps, using
/// colourful escape codes, severity categories and format strings.
class Log {
public:
    /// delete the copy constructor
    Log(Log& other) = delete;
    /// delete the copy constructor
    Log(const Log&) = delete;
    /// @brief delete the assignment operator
    void operator=(const Log&) = delete;

    ///@brief Log some debug message with an "INFO" tag
    ///
    ///@tparam Args
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void Info(std::format_string<Args...> fmt, Args&&... args)
    {
        _log("INFO", Log::ESC_CYAN, Log::ESC_CYAN_BOLD,
            std::format(fmt, std::forward<Args>(args)...));
    };

    ///@brief Log some debug message with an "INFO" tag
    ///
    ///@tparam Args
    ///@param tag optional coloured name to display before the message
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void InfoTagged(
        std::string_view tag, std::format_string<Args...> fmt, Args&&... args)
    {
        _log("INFO", Log::ESC_CYAN, Log::ESC_CYAN_BOLD,
            std::format(fmt, std::forward<Args>(args)...), tag);
    };

    ///@brief Log a warning with an "WARN" tag
    ///
    ///@tparam Args
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void Warn(std::format_string<Args...> fmt, Args&&... args)
    {
        _log("WARN", Log::ESC_YELLOW, Log::ESC_YELLOW_BOLD,
            std::format(fmt, std::forward<Args>(args)...));
    };

    ///@brief Log some debug message with an "WARN" tag
    ///
    ///@tparam Args
    ///@param tag optional coloured name to display before the message
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void WarnTagged(
        std::string_view tag, std::format_string<Args...> fmt, Args&&... args)
    {
        _log("WARN", Log::ESC_YELLOW, Log::ESC_YELLOW_BOLD,
            std::format(fmt, std::forward<Args>(args)...), tag);
    };

    ///@brief Log a success message with a "SUCC" tag
    ///
    ///@tparam Args
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void Success(std::format_string<Args...> fmt, Args&&... args)
    {
        _log("SUCC", Log::ESC_GREEN, Log::ESC_GREEN_BOLD,
            std::format(fmt, std::forward<Args>(args)...));
    };

    ///@brief Log some debug message with an "SUCC" tag
    ///
    ///@tparam Args
    ///@param tag optional coloured name to display before the message
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void SuccessTagged(
        std::string_view tag, std::format_string<Args...> fmt, Args&&... args)
    {
        _log("SUCC", Log::ESC_GREEN, Log::ESC_GREEN_BOLD,
            std::format(fmt, std::forward<Args>(args)...), tag);
    };

    ///@brief Log an error message with an "ERROR" tag
    ///
    ///@tparam Args
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void Error(std::format_string<Args...> fmt, Args&&... args)
    {
        _log("ERROR", Log::ESC_RED, Log::ESC_RED_BOLD,
            std::format(fmt, std::forward<Args>(args)...));
    };
    ///@brief Log some debug message with an "ERROR" tag
    ///
    ///@tparam Args
    ///@param tag optional coloured name to display before the message
    ///@param fmt `std::format` format string for log message
    ///@param args `std::format` arguments for log message
    template <typename... Args>
    static void ErrorTagged(
        std::string_view tag, std::format_string<Args...> fmt, Args&&... args)
    {
        _log("SUCC", Log::ESC_RED, Log::ESC_RED_BOLD,
            std::format(fmt, std::forward<Args>(args)...), tag);
    };

    /// @brief From this call onwards, stop logging until it is resumed with
    /// `start_logging`
    static void stop_logging()
    {
        Log& inst = instance();
        inst.logging = false;
    };
    /// @brief From this call onwards, start logging again after it was stopped
    /// with `stop_logging`
    static void start_logging()
    {
        Log& inst = instance();
        inst.logging = true;
    };

    ///@brief Flush the output stream in the destructor
    ~Log() { std::cout << std::flush; }

private:
    /// @brief Whether or not messages are currently being logged or simply
    /// ignored
    bool logging { true };

    /// @brief internal function used to implement the various public interfaces
    /// for the logger, since they share the same common structure: print some
    /// tag and a timestamp in some colour, then format the message and write it
    /// to an output stream
    /// @param name name tag to display in the message, like "INFO", "WARN" etc.
    /// @param colour ANSII escape sequence for the colour used to display the
    /// message tag
    /// @param bold ANSII escape sequence for the BOLD colour used to display
    /// the message tag
    /// @param msg the message to log
    /// @param tag optional coloured tag to display before the message
    static void _log(const char* name, const std::string_view colour,
        const std::string_view bold, std::string msg, std::string_view tag = "")
    {
        Log& inst = instance();
        if (!inst.logging)
            return;

        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        std::time_t t = system_clock::to_time_t(now);
        char hms_str[20];
        strftime(hms_str, sizeof(hms_str), "%T", std::localtime(&t));
        std::cout << std::format("{}[{}{}{} {}/{:03}{}] {} {} {}\n", colour,
            bold, name, Log::ESC_DIMMED, hms_str, static_cast<int>(ms.count()),
            colour, tag, Log::ESC_DEFAULT, msg);
    }

    // escape sequences taken from:
    // https://web.archive.org/web/20251118165851/https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797

    /// @brief ANSI Escape sequence for default foreground
    inline static constexpr std::string_view ESC_DEFAULT { "\x1b[39m" };
    /// @brief ANSI Escape sequence for dimmed foreground
    inline static constexpr std::string_view ESC_DIMMED { "\x1b[2;39m" };
    /// @brief ANSI Escape sequence for cyan foreground
    inline static constexpr std::string_view ESC_CYAN { "\x1b[36m" };
    /// @brief ANSI Escape sequence for bold cyan foreground
    inline static constexpr std::string_view ESC_CYAN_BOLD { "\x1b[1;36m" };
    /// @brief ANSI Escape sequence for yellow foreground
    inline static constexpr std::string_view ESC_YELLOW { "\x1b[33m" };
    /// @brief ANSI Escape sequence for bold yellow foreground
    inline static constexpr std::string_view ESC_YELLOW_BOLD { "\x1b[1;33m" };
    /// @brief ANSI Escape sequence for green foreground
    inline static constexpr std::string_view ESC_GREEN { "\x1b[32m" };
    /// @brief ANSI Escape sequence for bold green foreground
    inline static constexpr std::string_view ESC_GREEN_BOLD { "\x1b[1;32m" };
    /// @brief ANSI Escape sequence for red foreground
    inline static constexpr std::string_view ESC_RED { "\x1b[31m" };
    /// @brief ANSI Escape sequence for bold red foreground
    inline static constexpr std::string_view ESC_RED_BOLD { "\x1b[1;31m" };

    ///@brief The constructor does nothing and is protected, as intended in the
    /// Singleton pattern
    Log() { }

    ///@brief Obtain a pointer to the singleton log instance. This is used
    /// instead of a constructor, where a use of a `static` variable ensures
    /// only one singleton lives.
    ///@return Log& access to the singleton Log instance
    static Log& instance()
    {
        static Log instance;
        return instance;
    }
};

#endif // LOG_H_
