'''java
package com.example.mvc01.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

//使用Controller定义bean
@RestController //标记这个类是一个控制器类，可以处理HTTP请求，并且返回JSON或者XML格式的数据
@RequestMapping("/books") //标记这个类的所有方法的请求路径都是以"/books"开头的
public class UserController {
    @GetMapping("/{id}") //标记这个方法是用来处理GET请求，并且请求路径是"/books/{id}
    public String getById(@PathVariable Integer id){
        System.out.println("id= "+id);
        return "hello";
    }
}
'''
