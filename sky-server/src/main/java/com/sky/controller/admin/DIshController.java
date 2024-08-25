package com.sky.controller.admin;

import com.sky.dto.DishDTO;
import com.sky.result.Result;
import com.sky.service.DishService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Description:
 * author kano
 * create 2024-08-25
 * project IntelliJ IDEA
 */
@RestController
@RequestMapping("/admin/dish")
@Api(tags = "彩屏相关接口")
@Slf4j
public class DIshController {
    @Autowired
    private DishService dishService;
    @PostMapping()
    @ApiOperation("新增菜品")
    public Result save(@RequestBody DishDTO dto){
        log.info("新增菜品：{}",dto);
        dishService.saveWithFlavor(dto);
        return Result.success();
    }
}
