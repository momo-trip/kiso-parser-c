// include_analyzer.cpp
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileEntry.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <mutex>
#include <unistd.h>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #error "No filesystem support"
#endif

using namespace clang;
using namespace clang::tooling;

static const std::vector<std::string> DEFAULT_INCLUDE_PATHS = {
    "-resource-dir=/usr/lib/llvm-19/lib/clang/19",
    "-w",
    "-Wno-incompatible-function-pointer-types", 
    "-Wno-incompatible-pointer-types",
    "-isystem/usr/include/c++/11",
    "-isystem/usr/include/x86_64-linux-gnu/c++/11",
    "-isystem/usr/include/c++/11/backward",
    "-isystem/usr/include/x86_64-linux-gnu",
    "-isystem/usr/include",
    "-fno-strict-aliasing",
};

struct IncludeEntry {
    std::string including_file;
    std::string included_file; 
    std::string included_name;
    int line;
    int column;
    bool is_angled;                 // <> or ""
    bool is_system_header;
    std::string resolved_path; 
};

static std::vector<IncludeEntry> globalIncludes;
static std::mutex includesMutex;

class IncludeAnalyzer {
public:
    SourceManager *SM;
    
    explicit IncludeAnalyzer(SourceManager *SM) : SM(SM) {}
    
    std::string getAbsolutePath(StringRef filename) {
        fs::path p(filename.str());
        std::error_code ec;
        if (!p.is_absolute()) {
            p = fs::absolute(p, ec);
        }
        return p.lexically_normal().string();
    }
    
    std::string getFilePath(SourceLocation Loc) {
        if (Loc.isInvalid()) return "";
        
        SourceLocation OriginalLoc = Loc;
        if (Loc.isMacroID()) {
            OriginalLoc = SM->getSpellingLoc(Loc);
        }
        
        FileID FID = SM->getFileID(OriginalLoc);
        OptionalFileEntryRef FER = SM->getFileEntryRefForID(FID);
        if (!FER) return "";
        
        return getAbsolutePath(FER->getName());
    }
    
    int getLineNumber(SourceLocation Loc) {
        if (Loc.isInvalid()) return 0;
        
        SourceLocation OriginalLoc = Loc;
        if (Loc.isMacroID()) {
            OriginalLoc = SM->getSpellingLoc(Loc);
        }
        
        FileID FID = SM->getFileID(OriginalLoc);
        return SM->getLineNumber(FID, SM->getFileOffset(OriginalLoc));
    }
    
    int getColumnNumber(SourceLocation Loc) {
        if (Loc.isInvalid()) return 0;
        
        SourceLocation OriginalLoc = Loc;
        if (Loc.isMacroID()) {
            OriginalLoc = SM->getSpellingLoc(Loc);
        }
        
        FileID FID = SM->getFileID(OriginalLoc);
        return SM->getColumnNumber(FID, SM->getFileOffset(OriginalLoc));
    }
    
    bool isSystemHeader(SourceLocation Loc) {
        if (Loc.isInvalid()) return true;
        return SM->isInSystemHeader(Loc);
    }
};

class IncludeCallback : public PPCallbacks {
private:
    IncludeAnalyzer &Analyzer;
    Preprocessor &PP;

public:
    IncludeCallback(IncludeAnalyzer &Analyzer, Preprocessor &PP) 
        : Analyzer(Analyzer), PP(PP) {}

    void InclusionDirective(SourceLocation HashLoc,
                            const Token &IncludeTok,
                            StringRef FileName,
                            bool IsAngled,
                            CharSourceRange FilenameRange,
                            OptionalFileEntryRef File,
                            StringRef SearchPath,
                            StringRef RelativePath,
                            const Module *SuggestedModule,
                            bool ModuleImported,
                            SrcMgr::CharacteristicKind FileType) override {
        
        IncludeEntry entry;
        entry.including_file = Analyzer.getFilePath(HashLoc);
        entry.included_name = FileName.str();
        entry.line = Analyzer.getLineNumber(HashLoc);
        entry.column = Analyzer.getColumnNumber(HashLoc);
        entry.is_angled = IsAngled;
        entry.is_system_header = (FileType == SrcMgr::C_System || 
                                   FileType == SrcMgr::C_ExternCSystem);
        
        if (File) {
            entry.resolved_path = Analyzer.getAbsolutePath(File->getName());
            entry.included_file = entry.resolved_path;
        } else {
            entry.included_file = FileName.str();
            entry.resolved_path = "";
        }
        
        std::lock_guard<std::mutex> lock(includesMutex);
        globalIncludes.push_back(entry);
        
        llvm::errs() << "[Include] " << entry.including_file 
                     << ":" << entry.line << ":" << entry.column
                     << " -> " << (IsAngled ? "<" : "\"") 
                     << FileName.str() 
                     << (IsAngled ? ">" : "\"") << "\n";
    }
    
    void FileSkipped(const FileEntryRef &SkippedFile,
                     const Token &FilenameTok,
                     SrcMgr::CharacteristicKind FileType) override {
        // It is also possible to record files that were already included and skipped
        llvm::errs() << "[Skipped] " << SkippedFile.getName().str() 
                     << " (already included)\n";
    }
};

class IncludeASTConsumer : public ASTConsumer {
private:
    IncludeAnalyzer Analyzer;
    Preprocessor &PP;

public:
    explicit IncludeASTConsumer(SourceManager *SM, Preprocessor &PP) 
        : Analyzer(SM), PP(PP) {
        PP.addPPCallbacks(std::make_unique<IncludeCallback>(Analyzer, PP));
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        llvm::errs() << "Translation unit processed\n";
    }
};

class IncludeAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        return std::make_unique<IncludeASTConsumer>(
            &CI.getSourceManager(), 
            CI.getPreprocessor()
        );
    }
};

std::string escapeJSON(const std::string &s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c;
        }
    }
    return result;
}

void printIncludesAsJSON() {
    std::cout << "{\n";
    std::cout << "  \"includes\": [\n";
    
    bool first = true;
    for (const auto &entry : globalIncludes) {
        if (!first) {
            std::cout << ",\n";
        }
        first = false;
        
        std::cout << "    {\n";
        std::cout << "      \"including_file\": \"" << escapeJSON(entry.including_file) << "\",\n";
        std::cout << "      \"included_name\": \"" << escapeJSON(entry.included_name) << "\",\n";
        std::cout << "      \"included_file\": \"" << escapeJSON(entry.included_file) << "\",\n";
        std::cout << "      \"resolved_path\": \"" << escapeJSON(entry.resolved_path) << "\",\n";
        std::cout << "      \"line\": " << entry.line << ",\n";
        std::cout << "      \"column\": " << entry.column << ",\n";
        std::cout << "      \"is_angled\": " << (entry.is_angled ? "true" : "false") << ",\n";
        std::cout << "      \"is_system_header\": " << (entry.is_system_header ? "true" : "false") << "\n";
        std::cout << "    }";
    }
    
    std::cout << "\n  ]\n";
    std::cout << "}\n";
}

static llvm::cl::OptionCategory MyToolCategory("include analyzer options");

int main(int argc, const char **argv) {
    // Usage 1: ./include_analyzer source.c [-- extra_flags]
    if (argc >= 2 && std::string(argv[1]) != "-p") {
        std::vector<std::string> SourcePaths;
        SourcePaths.push_back(argv[1]);
        
        std::vector<std::string> CompileCommands;
        CompileCommands.insert(
            CompileCommands.end(),
            DEFAULT_INCLUDE_PATHS.begin(),
            DEFAULT_INCLUDE_PATHS.end()
        );
        
        bool afterDashes = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--") {
                afterDashes = true;
                continue;
            }
            if (afterDashes) {
                CompileCommands.push_back(argv[i]);
            }
        }
        
        auto Compilations = std::make_unique<FixedCompilationDatabase>(
            ".", CompileCommands);
        
        ClangTool Tool(*Compilations, SourcePaths);
        int result = Tool.run(newFrontendActionFactory<IncludeAction>().get());
        
        printIncludesAsJSON();
        return result;
    }
    
    // Usage 2: ./include_analyzer -p /path/to/compile_commands_dir
    if (argc >= 3 && std::string(argv[1]) == "-p") {
        std::string dirPath = argv[2];
        
        std::string ErrorMessage;
        auto Compilations = CompilationDatabase::autoDetectFromDirectory(dirPath, ErrorMessage);
        
        if (!Compilations) {
            llvm::errs() << "Error: " << ErrorMessage << "\n";
            return 1;
        }
        
        auto AllFiles = Compilations->getAllFiles();
        if (AllFiles.empty()) {
            llvm::errs() << "No source files found in compile_commands.json\n";
            return 1;
        }
        
        llvm::errs() << "Found " << AllFiles.size() << " source files\n";
        
        ClangTool Tool(*Compilations, AllFiles);
        int result = Tool.run(newFrontendActionFactory<IncludeAction>().get());
        
        printIncludesAsJSON();
        return result;
    }
    
    // Usage 2: ./include_analyzer -p /path/to/compile_commands_dir
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    
    int result = Tool.run(newFrontendActionFactory<IncludeAction>().get());
    printIncludesAsJSON();
    
    return result;
}