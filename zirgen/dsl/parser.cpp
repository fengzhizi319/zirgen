// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zirgen/dsl/parser.h"

namespace zirgen {
namespace dsl {

using namespace ast;
using std::make_shared;

// 获取二元操作符的优先级
BinaryOpPrecedence getPrecedence(Token token) {
  switch (token) {
  case tok_range:
    return 1; // 范围操作符的优先级
  case tok_bit_and:
    return 2; // 按位与操作符的优先级
  case tok_plus:
    return 3; // 加法操作符的优先级
  case tok_minus:
    return 3; // 减法操作符的优先级
  case tok_times:
    return 4; // 乘法操作符的优先级
  case tok_div:
    return 4; // 除法操作符的优先级
  case tok_mod:
    return 4; // 取模操作符的优先级
  case tok_dot:
    return 5; // 点操作符的优先级
  default:
    // 如果遇到非二元操作符，表示表达式结束，返回一个低于所有其他操作符的优先级
    return -1;
  }
}
// buildBinaryOp
// 函数的主要作用是根据给定的二元操作符（如加法、减法等）和操作数（左操作数和右操作数）
// 构建相应的表达式对象。该函数会根据操作符的类型选择合适的表达式构造函数，并将操作数传递给它。
Expression::Ptr
Parser::buildBinaryOp(Token token, Expression::Ptr&& lhs, Expression::Ptr&& rhs, SMLoc location) {
  std::string desugaredName;
  switch (token) {
  case tok_range:
    // 处理范围操作符
    return make_shared<Range>(location, std::move(lhs), std::move(rhs));
  case tok_bit_and:
    // 处理按位与操作符
    desugaredName = "BitAnd";
    break;
  case tok_mod:
    // 处理取模操作符
    desugaredName = "Mod";
    break;
  case tok_plus:
    // 处理加法操作符
    desugaredName = "Add";
    break;
  case tok_minus:
    // 处理减法操作符
    desugaredName = "Sub";
    break;
  case tok_times:
    // 处理乘法操作符
    desugaredName = "Mul";
    break;
  case tok_div:
    // 处理除法操作符
    desugaredName = "Div";
    break;
  default:
    // 处理未定义的二元操作符
    error("Undefined binary operator");
    desugaredName = "Undefined";
    break;
  }

  // 创建标识符组件
  Ident::Ptr component = make_shared<Ident>(location, desugaredName);
  // 创建参数列表
  Expression::Vec args;
  args.push_back(std::move(lhs));
  args.push_back(std::move(rhs));
  // 返回构造的表达式
  return make_shared<Construct>(location, std::move(component), std::move(args));
}
/*
lexer（词法分析器）的主要功能是将源代码转换为一系列标记（tokens）。这些标记是源代码的基本组成部分，每个标记代表一个最小的语法单元，如关键字、标识符、操作符、字面量等。词法分析器通过扫描源代码，识别并分类这些语法单元，从而为后续的语法分析（parser）提供输入。
主要功能
扫描源代码：逐字符读取源代码，识别出不同的语法单元。
生成标记：将识别出的语法单元转换为标记，每个标记包含类型和相关的值（如标识符的名称或字面量的值）。
跳过空白和注释：忽略源代码中的空白字符和注释，这些内容对语法分析没有影响。
错误处理：在遇到无法识别的字符或非法的语法单元时，记录错误信息。
主要方法
peekToken()：查看当前的标记，而不消耗它。
takeToken()：获取当前的标记，并移动到下一个标记。
takeTokenIf(tokenType)：如果当前标记是指定类型，则获取该标记并移动到下一个标记。
getIdentifier()：获取当前标记的标识符名称。
getLiteral()：获取当前标记的字面量值。
doImport(name)：处理导入语句。
通过这些功能，lexer 将源代码转换为标记流，为语法分析器提供结构化
*/
// 解析模块
Module::Ptr Parser::parseModule() {
  Component::Vec components; // 存储模块中的组件

  bool done = false; // 标记是否完成解析
  while (!done) {
    switch (lexer.peekToken()) { // 查看下一个标记
    case tok_hash:
    case tok_argument:
    case tok_component:
    case tok_function:
    case tok_extern:
      components.push_back(parseComponent()); // 解析组件并添加到组件列表
      break;
    case tok_import:
      parseImport(); // 解析导入语句
      break;
    case tok_test:
    case tok_test_fails: {
      Component::Ptr testMain = parseTest(); // 解析测试组件
      if (lexer.inMain()) {
        components.push_back(testMain); // 如果在主模块中，添加测试组件
      }
    } break;
    default:
      done = true; // 结束解析
      break;
    }
  }

  Token lastToken = lexer.takeToken(); // 获取最后一个标记
  if (lastToken != tok_eof) {
    error("unexpected input at end of file"); // 如果不是文件结束标记，报错
  }

  if (errors.empty()) {
    return make_shared<Module>(lexer.getLastLocation(), std::move(components)); // 返回解析的模块
  } else {
    return nullptr; // 如果有错误，返回空指针
  }
}

// 解析可选的属性列表
Attribute::Vec Parser::parseOptionalAttributeList() {
  if (lexer.peekToken() != tok_hash)
    return {};       // 如果下一个标记不是 '#', 返回空列表
  lexer.takeToken(); // 消耗 '#'

  if (!lexer.takeTokenIf(tok_square_l)) {
    error("Expected '#[' at the beginning of an attribute list"); // 如果不是 '[', 报错
    return {};
  }

  Attribute::Vec attributes; // 存储属性

  // 处理没有属性的情况
  if (lexer.takeTokenIf(tok_paren_r)) {
    return attributes; // 如果是 ']', 返回空属性列表
  }

  Token token;
  do {
    if (lexer.takeTokenIf(tok_ident)) {
      auto attribute =
          make_shared<Attribute>(lexer.getLastLocation(), lexer.getIdentifier()); // 创建属性
      attributes.push_back(attribute);                                            // 添加属性到列表
      token = lexer.takeToken();                                                  // 获取下一个标记
    } else {
      break; // 如果不是标识符，结束循环
    }
  } while (token == tok_comma && errors.empty()); // 如果是逗号，继续解析下一个属性

  if (token != tok_square_r) {
    error("Expected ']' at the end of an attribute list"); // 如果不是 ']', 报错
    return {};
  }

  return attributes; // 返回解析的属性列表
}

Component::Ptr Parser::parseComponent() {
  // 解析可选的属性列表
  Attribute::Vec attributes = parseOptionalAttributeList();

  ast::Component::Kind kind;
  // 根据标记类型确定组件的种类
  switch (lexer.takeToken()) {
  case tok_argument:
    kind = ast::Component::Kind::Argument; // 参数组件
    break;
  case tok_component:
    kind = ast::Component::Kind::Object; // 对象组件
    break;
  case tok_function:
    kind = ast::Component::Kind::Function; // 函数组件
    break;
  case tok_extern:
    return parseExtern(); // 外部组件
  default:
    // 如果没有属性且标记不匹配，报错
    if (attributes.empty()) {
      error("A component declaration must start with the `component` keyword");
    } else {
      error("Attributes are only allowed on component declarations");
    }
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation(); // 获取当前位置

  // 检查下一个标记是否是标识符
  if (lexer.takeToken() != tok_ident) {
    error("Expected an identifier name, got \"" + lexer.getIdentifier() + "\" instead");
    return nullptr;
  }
  std::string name = lexer.getIdentifier(); // 获取标识符名称

  // 解析可选的类型参数列表
  Parameter::Vec typeParams = parseOptionalTypeParameters();
  // 解析参数列表
  Parameter::Vec params = parseParameters();
  // 解析组件体
  Expression::Ptr body = parseBlock();

  // 创建并返回组件对象
  return make_shared<Component>(location,
                                kind,
                                name,
                                std::move(attributes),
                                std::move(typeParams),
                                std::move(params),
                                std::move(body));
}

Component::Ptr Parser::parseExtern() {
  SMLoc location = lexer.getLastLocation(); // 获取当前位置

  // 检查下一个标记是否是标识符
  if (!lexer.takeTokenIf(tok_ident)) {
    error("Expected an identifier name, got \"" + lexer.getIdentifier() + "\" instead");
    return nullptr;
  }
  std::string name = lexer.getIdentifier(); // 获取标识符名称

  Parameter::Vec typeParams; // 类型参数列表
  // 解析参数列表
  Parameter::Vec params = parseParameters();

  Expression::Ptr returnType; // 返回类型
  // 根据下一个标记确定返回类型
  switch (lexer.takeToken()) {
  case tok_semicolon: {
    // 默认返回类型为 Component
    returnType = make_shared<Ident>(location, "Component");
    break;
  }
  case tok_colon:
    // 解析返回类型表达式
    returnType = matchExpression("Expecting return type");
    if (!lexer.takeTokenIf(tok_semicolon)) {
      error("Expecting extern definition to be followed by a semicolon");
      return nullptr;
    }
    break;
  default:
    error("Expecting extern definition to end with either ';' or ': type;'");
    return nullptr;
  }

  // 创建并返回外部组件对象
  return make_shared<Component>(location,
                                Component::Kind::Extern,
                                name,
                                Attribute::Vec(),
                                std::move(typeParams),
                                std::move(params),
                                std::move(returnType));
}

Component::Ptr Parser::parseTest() {
  SMLoc location = lexer.getLastLocation();            // 获取当前位置
  bool isFail = (lexer.takeToken() == tok_test_fails); // 判断是否是失败测试

  std::string name;
  if (lexer.takeTokenIf(tok_ident)) {
    name = lexer.getIdentifier(); // 获取测试名称
  } else {
    name = std::to_string(testNum++); // 如果没有名称，使用测试编号
  }

  Expression::Ptr body = parseBlock();                             // 解析测试体
  name = std::string(isFail ? "test$fail$" : "test$succ$") + name; // 根据测试结果前缀名称

  // 创建并返回测试组件对象
  return make_shared<Component>(location,
                                ast::Component::Kind::Object,
                                name,
                                Attribute::Vec(),
                                Parameter::Vec(),
                                Parameter::Vec(),
                                body);
}

Block::Ptr Parser::parseBlock() {
  if (lexer.takeToken() != tok_curly_l) {
    error("A block must start with '{'"); // 块必须以 '{' 开始
    return nullptr;
  }
  Statement::Vec body;                      // 存储块中的语句
  Expression::Ptr lastExpression;           // 存储最后一个表达式
  SMLoc blockLoc = lexer.getLastLocation(); // 获取块的位置

  bool done = false; // 标记是否完成解析
  while (!done) {
    Token token = lexer.peekToken();          // 查看下一个标记
    SMLoc location = lexer.getLastLocation(); // 获取当前位置
    Access upcomingAccess = Access::Default;  // 默认访问权限
    switch (token) {
    case tok_curly_r:
      done = true; // 结束解析
      break;
    case tok_global:
    case tok_public:
      lexer.takeToken(); // 消耗标记
      if (token == tok_global) {
        upcomingAccess = Access::Global; // 全局访问权限
      } else {
        upcomingAccess = Access::Public; // 公共访问权限
      }
      // Fallthrough
    default:
      // 解析定义、声明、约束、void 或值
      lastExpression = matchExpression("expected expression at beginning of statement");

      if (lexer.peekToken() == tok_semicolon) {
        // void 语句
        lexer.takeToken(); // 消耗分号
        body.push_back(
            make_shared<Void>(location, std::move(lastExpression))); // 添加 void 语句到块中
        lastExpression = nullptr;
      } else if (lexer.peekToken() == tok_bang) {
        // 编译器指令
        location = lexer.getLastLocation();
        std::string name = lexer.getIdentifier();
        lexer.takeToken(); // 消耗 '!'
        if (!lexer.takeTokenIf(tok_paren_l)) {
          error("expected '(' after '!' in compiler directive"); // 指令后应有 '('
        }
        Expression::Vec arguments; // 存储指令参数
        if (lexer.peekToken() != tok_paren_r) {
          arguments = parseExpressions(); // 解析参数
        }
        lexer.takeToken(); // 消耗 ')'
        if (lexer.takeToken() != tok_semicolon) {
          error("expected semicolon after compiler directive"); // 指令后���有分号
          return nullptr;
        }

        body.push_back(
            make_shared<Directive>(location, name, std::move(arguments))); // 添加指令到块中
        lastExpression = nullptr;
      } else if (lexer.peekToken() == tok_define) {
        // 定义语句
        if (!lastExpression || !Ident::classof(lastExpression.get())) {
          error("expected an identifier on left side of member definition"); // 定义左侧应为标识符
          return nullptr;
        }
        auto identifier = dynamic_cast<Ident*>(lastExpression.get())->getName();

        lexer.takeToken(); // 消耗 'define'
        location = lexer.getLastLocation();
        Expression::Ptr defBody = matchExpression(
            "expected an expression on right side of member definition"); // 解析定义体
        if (lexer.takeToken() != tok_semicolon) {
          error("expected semicolon after member definition"); // 定义后应有分号
          return nullptr;
        }
        body.push_back(make_shared<Definition>(
            location, identifier, std::move(defBody), upcomingAccess)); // 添加定义到块中
        upcomingAccess = Access::Default;                               // 重置访问权限
        lastExpression = nullptr;
      } else if (lexer.peekToken() == tok_eq) {
        // 约束语句
        lexer.takeToken(); // ���耗 '='
        location = lexer.getLastLocation();
        Expression::Ptr rhs =
            matchExpression("expected an expression on right side of constraint"); // 解析右侧表达式
        if (lexer.takeToken() != tok_semicolon) {
          error("expected semicolon after constraint statement"); // 约束后应有分号
          return nullptr;
        }
        body.push_back(make_shared<Constraint>(
            location, std::move(lastExpression), std::move(rhs))); // 添加约束到块中
        lastExpression = nullptr;
      } else if (lexer.peekToken() == tok_colon) {
        // 声明语句
        if (!lastExpression || !Ident::classof(lastExpression.get())) {
          error("expected an identifier on left side of member declaration"); // 声明左侧应为标识符
          return nullptr;
        }
        auto identifier = dynamic_cast<Ident*>(lastExpression.get())->getName();

        lexer.takeToken(); // 消耗 ':'
        location = lexer.getLastLocation();
        Expression::Ptr declType = matchExpression("expected type expression"); // 解析类型表达式
        if (lexer.takeToken() != tok_semicolon) {
          error("expected semicolon after member declaration"); // 声明后应有分号
          return nullptr;
        }
        body.push_back(make_shared<Declaration>(
            location, identifier, std::move(declType), upcomingAccess)); // 添加声明到块中
        upcomingAccess = Access::Default;                                // 重置访问权限
        lastExpression = nullptr;
      } else if (lexer.peekToken() != tok_curly_r) {
        error("invalid statement"); // 无效语句
        return nullptr;
      }
      switch (upcomingAccess) {
      case Access::Global:
        error("Expected declaration or definition after `global'"); // `global` 后应���声明或定义
        return nullptr;
      case Access::Public:
        error("Expected declaration or definition after `public'"); // `public` 后应为声明或定义
        return nullptr;
      case Access::Default:
        break;
      }
      break;
    }
  }

  if (lexer.takeToken() != tok_curly_r) {
    error("A block must end with '}'"); // 块必须以 '}' 结束
    return nullptr;
  }

  if (!lastExpression) {
    Ident::Ptr component = make_shared<Ident>(blockLoc, "Component"); // 创建默认组件
    Expression::Vec args;
    lastExpression =
        make_shared<Construct>(blockLoc, std::move(component), std::move(args)); // 创建构造表达式
  }

  if (errors.empty()) {
    return make_shared<Block>(blockLoc, std::move(body), std::move(lastExpression)); // 返回解析的块
  } else {
    return nullptr; // 如果有错误，返回空指针
  }
}

void Parser::parseImport() {
  // 导入语句必须是 'import <ident>;' 的形式
  lexer.takeToken(); // 消耗 'import' 标记
  if (!lexer.takeTokenIf(tok_ident)) {
    error("导入语句中需要标识符"); // 如果没有标识符，报错
    return;
  }
  std::string name = lexer.getIdentifier(); // 获取标识符名称
  if (!lexer.takeTokenIf(tok_semicolon)) {
    error("导入语句后需要分号"); // 如果没有分号，报错
    return;
  }
  lexer.doImport(name); // 处理导入
}

Parameter::Vec Parser::parseOptionalTypeParameters() {
  if (!lexer.takeTokenIf(tok_angle_l)) {
    // 如果没有类型参数，不报错
    return {};
  }

  Parameter::Vec typeParams; // 存储类型参数
  Token token;
  do {
    Parameter::Ptr param = parseParameter(); // 解析参数
    token = lexer.takeToken();               // 获取下一个标记
    if (param) {
      typeParams.push_back(std::move(param)); // 添加参数到列表
    } else {
      break;
    }
  } while (token == tok_comma && errors.empty()); // 如果是逗号，继续解析下一个参数

  if (token != tok_angle_r) {
    error("类型参数列表末尾缺少 '>'"); // 如果不是 '>', 报错
    return {};
  }

  return typeParams; // 返回解析的类型参数列表
}

Parameter::Vec Parser::parseParameters() {
  if (!lexer.takeTokenIf(tok_paren_l)) {
    error("参数列表开头需要 '('"); // 参数列表开头需要 '('
    return {};
  }

  // 处理没有参数的情况
  if (lexer.takeTokenIf(tok_paren_r)) {
    return {};
  }

  Parameter::Vec params; // 存储参数
  Token token;
  do {
    Parameter::Ptr param = parseParameter(); // 解析参数
    token = lexer.takeToken();               // 获取下一个标记
    if (param) {
      params.push_back(std::move(param)); // 添加参数到列表
    } else {
      break;
    }
  } while (token == tok_comma && errors.empty()); // 如果是逗号，继续解析下一个参数

  if (token != tok_paren_r) {
    error("参数列表末尾需要 ')'"); // 参数列表末尾需要 ')'
    return {};
  }

  return params; // 返回解析的参数列表
}

Parameter::Ptr Parser::parseParameter() {
  Token token = lexer.takeToken(); // 获取下一个标记
  if (token != tok_ident) {
    error("参数必须以名称开头"); // 参数必须以名称开头
    return nullptr;
  }
  std::string name = lexer.getIdentifier(); // 获取参数名称
  SMLoc location = lexer.getLastLocation(); // 获取当前位置
  if (lexer.takeToken() != tok_colon) {
    error("参数名称后需要类型注释 ': type'"); // 参数名称后需要类型注释 ': type'
    return nullptr;
  }
  Expression::Ptr type = matchExpression("参数名称后需要类型注释表达式"); // 解析类型注释表达式
  if (!type) {
    return nullptr;
  }
  token = lexer.peekToken();                 // 查看下一个标记
  bool isVariadic = (token == tok_variadic); // 判断是否是可变参数
  if (isVariadic) {
    lexer.takeToken(); // 消耗可变参数标记
  }

  return make_shared<Parameter>(location, name, std::move(type), isVariadic); // 创建并返回参数对象
}

Expression::Ptr Parser::parseExpression(BinaryOpPrecedence precedence) {
  // 表达式语法是左递归的，所以我们使用自底向上的解析方法
  // 在任何时候，leftExpr 都保存了当前构造的最大的左侧子表达式
  Expression::Ptr leftExpr;

  bool done = false; // 标记是否完成解析
  while (!done) {
    Token token = lexer.peekToken(); // 查看下一个标记
    switch (token) {
    case tok_curly_l:
      // '{' 表示块的开始；块是一种表达式，但它也可以跟在另一个表达式之后
      // 例如在 map 的数组子表达式之后。这个保护措施防止将块解析为数组子表达式的一部分。
      if (leftExpr) {
        done = true;
        break;
      }
      [[fallthrough]];
    case tok_literal:
    case tok_string_literal:
    case tok_ident:
    case tok_for:
    case tok_reduce:
    case tok_if:
    case tok_square_l:
    case tok_angle_l:
    case tok_paren_l:
    case tok_mux:
    case tok_dot:
    case tok_back:
      if (leftExpr) {
        error("意外的表���式"); // 意外的表达式
      }
      leftExpr = parsePrimaryExpression(); // 解析主表达式
      break;

    // 中缀操作符
    case tok_range:
    case tok_bit_and:
    case tok_plus:
    case tok_times:
    case tok_div:
    case tok_mod:
      if (!leftExpr) {
        error("二元操作符缺少左操作数"); // 二元操作符缺少左操作数
      }
      leftExpr = parseBinaryOp(std::move(leftExpr), precedence); // 解析二元操作符
      break;

    // 前缀或中缀操作符
    case tok_minus:
      if (leftExpr) {
        leftExpr = parseBinaryOp(leftExpr, precedence); // 解析二元操作符
      } else {
        leftExpr = parseNegate(); // 解析取反操作符
      }
      break;

    default:
      // 遇到不属于表达式的内容，解析完成
      done = true;
    };
  }

  return leftExpr; // 返回解析的表达式
}

// 匹配表达式并返回解析结果
Expression::Ptr Parser::matchExpression(std::string message) {
  // 解析表达式
  auto expr = parseExpression();
  // 如果解析失败，记录错误信息
  if (!expr)
    error(message);
  // 返回解析的表达式
  return expr;
}

Expression::Ptr Parser::parsePrimaryExpression() {
  // Parse expressions which do not directly contain infix operators.
  // This subgrammar is left recursive, so we use bottom up parsing. At any
  // point in time, leftExpr holds the "largest" leftmost subexpression that
  // has been constructed yet.
  // 解析不直接包含中缀操作符的表达式。
  // 这个子语法是左递归的，所以我们使用自底向上的解析方法。
  // 在任何时候，leftExpr 都保存了当前构造的最大的左侧子表达式。
  Expression::Ptr leftExpr;

  bool done = false; // 标记是否完成解析
  while (!done) {
    switch (lexer.peekToken()) { // 查看下一个标记
    // non-left-recursive expressions
    // 非左递归表达式
    case tok_literal:
    case tok_string_literal:
      if (!leftExpr) {
        leftExpr = parseLiteral(); // 解析字面量
      } else {
        done = true;
      }
      break;
    case tok_ident:
      if (!leftExpr) {
        leftExpr = parseIdentifier(); // 解析标识符
      } else {
        done = true;
      }
      break;
    case tok_for:
      if (!leftExpr) {
        leftExpr = parseMap(); // 解析 for 循环
      } else {
        done = true;
      }
      break;
    case tok_reduce:
      if (!leftExpr) {
        leftExpr = parseReduce(); // 解析 reduce 表达式
      } else {
        done = true;
      }
      break;
    case tok_if:
      if (!leftExpr) {
        leftExpr = parseConditional(); // 解析条件表达式
      } else {
        done = true;
      }
      break;
    case tok_curly_l:
      // Sometimes a block may directly follow another primary expression that
      // it is not part of, for example in a map:
      //   for i : arr { ... }
      // 有时一个块可能直接跟在另一个��表达式之后，但它不是该表达式的一部分，例如在 map 中：
      //   for i : arr { ... }
      if (!leftExpr) {
        leftExpr = parseBlock(); // 解析块
      }
      done = true;
      break;
    case tok_minus:
      if (leftExpr) {
        done = true;
      } else {
        leftExpr = parseNegate(); // 解析取反操作符
      }
      break;

    // left-recursive expressions
    // 左递归表达式
    case tok_square_l:
      if (leftExpr) {
        leftExpr = parseSubscript(std::move(leftExpr)); // 解析下标
      } else {
        leftExpr = parseArrayLiteral(); // 解析数组字面量
      }
      break;
    case tok_angle_l:
      if (!leftExpr) {
        error("missing base expression for specialization"); // 缺少基础表达式用于特化
      }
      leftExpr = parseSpecialize(std::move(leftExpr)); // 解析特化表达式
      break;
    case tok_paren_l:
      if (leftExpr) {
        leftExpr = parseConstruct(std::move(leftExpr)); // 解析构造表达式
      } else {
        leftExpr = parseParenthesizedExpression(); // 解析括号表达式
      }
      break;
    case tok_mux:
      if (!leftExpr) {
        error("missing base expression for switch operation"); // 缺少基础表达式用于 switch 操作
      }
      leftExpr = parseSwitch(std::move(leftExpr)); // 解析 switch 表达式
      break;
    case tok_dot:
      if (!leftExpr) {
        error("missing base expression for member lookup"); // 缺少基础表达式用于成员查找
      }
      leftExpr = parseLookup(std::move(leftExpr)); // 解析成员查找
      break;
    case tok_back:
      if (!leftExpr) {
        error("missing base expression for back operation"); // 缺少基础表达式用于 back 操作
      }
      leftExpr = parseBack(std::move(leftExpr)); // 解析 back 表达式
      break;
    default:
      // Reached something that isn't part of the expression, so we're done
      // 遇到不属于表达式的内容，解析完成
      done = true;
      if (!leftExpr) {
        error("expected a primary expression here"); // 这里需要一个主表达式
      }
    }
  }

  return leftExpr; // 返回解析的表达式
}
/*
"literal" 在编程中通常指的是字面量，即在代码中直接表示固定值的常量。
例如，数字 42、字符串 "hello"、布尔值 true 都是字面量。
*/
Literal::Ptr Parser::parseLiteral() {
  Token token = lexer.takeToken();
  if (token == tok_literal) {
    // Create a Literal object with the current location and literal value
    // 使用当前位置和字面量值创建一个 Literal 对象
    return make_shared<Literal>(lexer.getLastLocation(), lexer.getLiteral());
  } else if (token == tok_string_literal) {
    // Create a StringLiteral object with the current location and identifier
    // 使用当前位置和标识符创建一个 StringLiteral 对象
    return make_shared<StringLiteral>(lexer.getLastLocation(), lexer.getIdentifier());
  } else {
    // Return nullptr if the token is not a literal or string literal
    // 如果标记不是字面���或字符串字面量，则返回空指针
    return nullptr;
  }
}

Ident::Ptr Parser::parseIdentifier() {
  if (lexer.takeToken() == tok_ident) {
    // Create an Ident object with the current location and identifier
    // 使用当前位置和标识符创建一个 Ident 对象
    return make_shared<Ident>(lexer.getLastLocation(), lexer.getIdentifier());
  } else {
    // Report an error if the token is not an identifier
    // 如果标记不是标识符，则报告错误
    error("An identifier must start with a letter");
    return nullptr;
  }
}

Map::Ptr Parser::parseMap() {
  if (lexer.takeToken() != tok_for) {
    // Report an error if the token is not 'for'
    // 如果标记不是 'for'，则报告错���
    error("A map construct must start with the keyword 'for'");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  if (lexer.takeToken() != tok_ident) {
    // Report an error if the token after 'for' is not an identifier
    // 如果 'for' 之后的标记不是标识符，则报告错误
    error("Expected an identifier after 'for'");
    return nullptr;
  }
  std::string inductionVar = lexer.getIdentifier();

  if (lexer.takeToken() != tok_colon) {
    // Report an error if the token after the induction variable is not ':'
    // 如果归纳变量之后的标记不是 ':'，则报告错误
    error("Expected ':' after induction variable in map construct");
    return nullptr;
  }

  // Parse the array expression
  // 解析数组表达式
  Expression::Ptr array = matchExpression("Expected array expression");
  // Parse the function block
  // 解析函数块
  Expression::Ptr function = parseBlock();

  // Create and return a Map object
  // 创建并返回一个 Map 对象
  return make_shared<Map>(location, std::move(array), inductionVar, std::move(function));
}

Reduce::Ptr Parser::parseReduce() {
  if (lexer.takeToken() != tok_reduce) {
    // Report an error if the token is not 'reduce'
    // 如果标记不是 'reduce'，则报告错误
    error("A reduce construct must start with the keyword 'reduce'");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  // Parse the array expression
  // 解析数组表达式
  Expression::Ptr array = matchExpression("Expected array expression");

  if (lexer.takeToken() != tok_init) {
    // Report an error if the token after the array is not 'init'
    // 如果数组之后的标记不是 'init'，则报告错误
    error("Expected initial value in reduce construct");
    return nullptr;
  }

  // Parse the initial value expression
  // 解析初始值表达式
  Expression::Ptr init = matchExpression("Expected init expression");

  if (lexer.takeToken() != tok_with) {
    // Report an error if the token after the initial value is not 'with'
    // 如果初始值之后的标记不是 'with'，则报告错误
    error("Expected a reducer component in reduce construct");
    return nullptr;
  }

  // Parse the reducer expression
  // 解析 reducer 表达式
  Expression::Ptr reducer = matchExpression("Expected reducer expression");

  // Create and return a Reduce object
  // 创建并返回一个 Reduce 对象
  return make_shared<Reduce>(location, std::move(array), std::move(init), std::move(reducer));
}

Switch::Ptr Parser::parseConditional() {
  if (!lexer.takeTokenIf(tok_if)) {
    // Report an error if the token is not 'if'
    // 如果标记不是 'if'，则报告错误
    error("A conditional expression must start with the 'if' keyword");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  if (!lexer.takeTokenIf(tok_paren_l)) {
    // Report an error if the token after 'if' is not '('
    // 如果 'if' 之后的标记不是 '('，则报告错误
    error("Expected parentheses around condition, missing '('");
    return nullptr;
  }

  // Parse the condition expression
  // 解析条件表达式
  Expression::Ptr condition = matchExpression("Expected a condition expression");

  if (!lexer.takeTokenIf(tok_paren_r)) {
    // Report an error if the token after the condition is not ')'
    // 如果条件之后的标记不是 ')'，则报告错误
    error("Missing ')' after condition");
    return nullptr;
  }

  Expression::Vec cases;
  Expression::Vec selectors;

  // Parse the 'if' block
  // 解析 'if' 块
  cases.push_back(parseBlock());
  selectors.push_back(condition);
  if (lexer.takeTokenIf(tok_else)) {
    // Parse the 'else' block
    // 解析 'else' 块
    cases.push_back(parseBlock());
    Expression::Vec subArgs;
    subArgs.push_back(make_shared<Literal>(location, 1));
    subArgs.push_back(condition);
    selectors.push_back(
        make_shared<Construct>(location, make_shared<Ident>(location, "Sub"), subArgs));
  }
  // Create an array literal for the selectors
  // 为选择器创建一个数组字面量
  Expression::Ptr selector = make_shared<ArrayLiteral>(location, std::move(selectors));
  // Create and return a Switch object
  // 创建并返回一个 Switch 对象
  return make_shared<Switch>(location, std::move(selector), std::move(cases), false);
}

Expression::Ptr Parser::parseParenthesizedExpression() {
  if (!lexer.takeTokenIf(tok_paren_l)) {
    // Report an error if the token is not '('
    // 如果标记不是 '('，则报告错误
    error("Expected a '(' at the start of a parenthesized expression");
    return nullptr;
  }

  // Parse the expression inside the parentheses
  // 解析括号内的表达式
  Expression::Ptr expression = parseExpression();

  if (!lexer.takeTokenIf(tok_paren_r)) {
    // Report an error if the token is not ')'
    // 如果标记不是 ')'，则报告错误
    error("Expected a ')' at the end of a parenthesized expression");
    return nullptr;
  }

  // Return the parsed expression
  // 返回解析的表达式
  return expression;
}

ArrayLiteral::Ptr Parser::parseArrayLiteral() {
  if (!lexer.takeTokenIf(tok_square_l)) {
    // Report an error if the token is not '['
    // 如果标记不是 '['，则报告错误
    error("Expected a '[' at the start of an array literal");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  Expression::Vec elements;
  if (lexer.peekToken() != tok_square_r) {
    // Parse the elements of the array
    // 解析数组的元素
    elements = parseExpressions();
  }

  if (!lexer.takeTokenIf(tok_square_r)) {
    // Report an error if the token is not ']'
    // 如果标记不是 ']'，则报告错误
    error("Expected a ']' at the end of an array literal");
    return nullptr;
  }

  // Create and return an ArrayLiteral object
  // 创建并返回一个 ArrayLiteral 对象
  return make_shared<ArrayLiteral>(location, std::move(elements));
}

Back::Ptr Parser::parseBack(Expression::Ptr&& baseExpr) {
  if (lexer.takeToken() != tok_back) {
    // Report an error if the token is not '@'
    // 如果标记不是 '@'，则报告错误
    error("Expected a '@' after base component in back expression");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();
  Expression::Ptr rhs;

  switch (lexer.peekToken()) {
  case tok_paren_l:
    // Parse a parenthesized expression
    // 解析括号表达式
    rhs = parseParenthesizedExpression();
    break;
  case tok_literal:
    // Parse a literal
    // 解析字面量
    rhs = parseLiteral();
    break;
  case tok_ident:
    // Parse an identifier
    // 解析标识符
    rhs = parseIdentifier();
    break;
  default:
    rhs = nullptr;
    break;
  }

  if (!rhs) {
    // Report an error if the back distance could not be parsed
    // 如果无法解析 back 距离，则报告错误
    error("Unable to parse back distance");
    return nullptr;
  }
  // Create and return a Back object
  // 创建并返回一个 Back 对象
  return make_shared<Back>(location, std::move(baseExpr), std::move(rhs));
}

Lookup::Ptr Parser::parseLookup(Expression::Ptr&& baseExpr) {
  if (lexer.takeToken() != tok_dot) {
    // Report an error if the token is not '.'
    // 如果标记不是 '.'，则报告错误
    error("Expected a '.' after base component in member lookup expression");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  if (lexer.takeToken() != tok_ident) {
    // Report an error if the token after '.' is not an identifier
    // 如果 '.' 之后的标记不是标识符，则报告错误
    error("Expected an identifier");
    return nullptr;
  }
  std::string member = lexer.getIdentifier();

  // Create and return a Lookup object
  // 创建并返回一个 Lookup 对象
  return make_shared<Lookup>(location, std::move(baseExpr), member);
}

Subscript::Ptr Parser::parseSubscript(Expression::Ptr&& primaryExpr) {
  if (!lexer.takeTokenIf(tok_square_l)) {
    // Report an error if the token is not '['
    // 如果标记不是 '['，则报告错误
    error("Expected a '[' after array component in subscript expression");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  // Parse the index expression
  // 解析索引表达式
  Expression::Ptr element = matchExpression("Expected index expression");

  if (!lexer.takeTokenIf(tok_square_r)) {
    // Report an error if the token is not ']'
    // 如果标记不是 ']'，则报告错误
    error("Expected a ']' after element of subscript expression");
    return nullptr;
  }

  // Create and return a Subscript object
  // 创建并返回一个 Subscript 对象
  return make_shared<Subscript>(location, std::move(primaryExpr), std::move(element));
}

Specialize::Ptr Parser::parseSpecialize(Expression::Ptr&& generic) {
  if (!lexer.takeTokenIf(tok_angle_l)) {
    // Report an error if the token is not '<'
    // 如果标记不是 '<'，则报告错误
    error("Expected a '<' after the type in a \"specialize\" expression");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  // Parse the type arguments
  // 解析类型参数
  Expression::Vec typeArguments = parseExpressions();

  if (!lexer.takeTokenIf(tok_angle_r)) {
    // Report an error if the token is not '>'
    // 如果标记不是 '>'，则报告错误
    error("Expected a '>' after type arguments in a \"specialize\" expression");
    return nullptr;
  }

  // Create and return a Specialize object
  // 创建并返回一个 Specialize 对象
  return make_shared<Specialize>(location, std::move(generic), std::move(typeArguments));
}

Construct::Ptr Parser::parseConstruct(Expression::Ptr&& component) {
  if (!lexer.takeTokenIf(tok_paren_l)) {
    // Report an error if the token is not '('
    // 如果标记不是 '('，则报告错误
    error("Expected a '(' after component type in constructor expression");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  Expression::Vec arguments;
  if (lexer.peekToken() != tok_paren_r) {
    // Parse the arguments
    // 解析参数
    arguments = parseExpressions();
  }

  if (!lexer.takeTokenIf(tok_paren_r)) {
    // Report an error if the token is not ')'
    // 如果标记不是 ')'，则报告错误
    error("Expected a ')' after arguments in constructor expression");
    return nullptr;
  }

  // Create and return a Construct object
  // 创建并返回一个 Construct 对象
  return make_shared<Construct>(location, std::move(component), std::move(arguments));
}

Switch::Ptr Parser::parseSwitch(Expression::Ptr&& selector) {
  // The selector has already been parsed
  // 选择器已经被解析
  if (!lexer.takeTokenIf(tok_mux)) {
    // Expected '->' after selector in mux
    // 在 mux 中选择器后面需要 '->'
    error("Expected '->' after selector in mux");
    return nullptr;
  }
  SMLoc location = lexer.getLastLocation();

  // If there is a bang, it's the major mux
  // 如果有感叹号，这是主要的 mux
  bool isMajor = lexer.takeTokenIf(tok_bang);

  if (!lexer.takeTokenIf(tok_paren_l)) {
    // A mux's arms should be enclosed in parentheses, missing '('
    // mux 的分支应该用括号括起来，缺少 '('
    error("A mux's arms should be enclosed in parentheses, missing '('");
    return nullptr;
  }

  if (lexer.peekToken() == tok_paren_r) {
    // A mux must have at least one arm
    // mux 必须至少有一个分支
    error("A mux must have at least one arm");
    return nullptr;
  }

  // Parse the cases of the mux
  // 解析 mux 的分支
  Expression::Vec cases = parseExpressions();

  if (!lexer.takeTokenIf(tok_paren_r)) {
    // Expected ')' after all of the mux's arms
    // 在所有 mux 的分支之后需要 ')'
    error("Expected ')' after all of the mux's arms");
    return nullptr;
  }

  // Create and return a Switch object
  // 创建并返回一个 Switch 对象
  return make_shared<Switch>(location, std::move(selector), std::move(cases), isMajor);
}

Expression::Ptr Parser::parseBinaryOp(Expression::Ptr lhs, BinaryOpPrecedence precedence) {
  while (true) {
    Token opToken = lexer.peekToken();
    SMLoc location = lexer.getLastLocation();
    BinaryOpPrecedence opPrecedence = getPrecedence(opToken);

    if (opPrecedence < precedence) {
      // The next operation binds less tightly than the "unresolved" one, so
      // now we can resolve it. For example, if we see "a * b + ...", then the
      // rhs of the multiply is b and doesn't involve the "+ ..." because + has
      // lower precedence than *.
      // 下一个操作符的绑定优先级低于当前的未解析操作符，所以我们现在可以解析它。
      // 例如，如果我们看到 "a * b + ..."，那么乘法的右操作数是 b，并且不涉及 "+ ..."，因为 +
      // 的优先级低于 *。
      return lhs;
    }

    lexer.takeToken();
    Expression::Ptr rhs = parsePrimaryExpression();

    Token nextOpToken = lexer.peekToken();
    BinaryOpPrecedence nextOpPrecedence = getPrecedence(nextOpToken);
    if (opPrecedence < nextOpPrecedence) {
      // If the operation after the rhs has higher precedence than the one
      // between lhs and rhs, then it is also part of the rhs. For example, if
      // we see "a + b * c" then the rhs of the add is "b * c" and not just "b"
      // because + has lower precedence than *.
      // 如果 rhs 之后的操作符优先级高于 lhs 和 rhs 之间的操作符，那么它也是 rhs 的一部分。
      // 例如，如果我们看到 "a + b * c"，那么加法的右操作数是 "b * c" 而不仅仅是 "b"，因为 +
      // 的优先级低于 *。
      rhs = parseBinaryOp(std::move(rhs), opPrecedence + 1);
    }

    lhs = buildBinaryOp(opToken, std::move(lhs), std::move(rhs), location);
  }
}

Expression::Vec Parser::parseExpressions() {
  // Comma-separated list of one or more expressions
  // 逗号分隔的一个或多个表达式列表
  Expression::Vec expressions;

  do {
    Expression::Ptr expression = parseExpression();
    if (expression) {
      expressions.push_back(std::move(expression));
    } else {
      // Expected an expression
      // 需要一个表达式
      error("Expected an expression");
      break;
    }
  } while (lexer.takeTokenIf(tok_comma));

  return expressions;
}

Expression::Ptr Parser::parseNegate() {
  if (!lexer.takeTokenIf(tok_minus)) {
    // expected minus sign at start of negation expression
    // 负号表达式开头需要负号
    error("expected minus sign at start of negation expression");
    return nullptr;
  }

  SMLoc location = lexer.getLastLocation();
  Expression::Ptr val = parsePrimaryExpression();
  if (!val) {
    // expected expression after minus sign
    // 负号后需要表达式
    error("expected expression after minus sign");
  }

  Ident::Ptr id = make_shared<Ident>(location, "Neg");
  Expression::Vec args;
  args.push_back(std::move(val));
  return make_shared<Construct>(location, std::move(id), std::move(args));
}

} // namespace dsl
} // namespace zirgen
