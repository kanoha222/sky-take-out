package com.sky.handler;

import com.sky.constant.MessageConstant;
import com.sky.exception.BaseException;
import com.sky.result.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.sql.SQLIntegrityConstraintViolationException;

/**
 * 全局异常处理器，处理项目中抛出的业务异常
 */
//一个组合注解，结合了 @ControllerAdvice 和 @ResponseBody，用于全局处理控制器层的异常。
// 它的作用是定义一个全局的异常处理器，这样当任何控制器方法抛出异常时，都可以在这里进行捕获和处理。
//    所有在控制器层抛出的异常都会在这里被捕获处理。
@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {

    /**
     * 捕获业务异常,当控制器方法抛出指定类型的异常时，Spring 会调用带有 @ExceptionHandler 注解的方法来处理这个异常。
     * @param ex
     * @return
     */
    @ExceptionHandler
    public Result exceptionHandler(BaseException ex){
        log.error("异常信息：{}", ex.getMessage());
        return Result.error(ex.getMessage());
    }

    /**
     * 处理sql异常
     * @param ex
     * @return
     */
    @ExceptionHandler
    public Result exceptionHandler(SQLIntegrityConstraintViolationException ex){
        //Duplicate entry 'kano' for key 'idx_username'] with root cause
        String errMessage = ex.getMessage();
        if (errMessage.contains("Duplicate entry")){
            String[] split = errMessage.split(" ");
            String username = split[2];
            String msg = username + MessageConstant.ALREADY_EXISTS;
            return Result.error(msg);
        }
        else {
            return Result.error(MessageConstant.UNKNOWN_ERROR);
        }
    }
}
